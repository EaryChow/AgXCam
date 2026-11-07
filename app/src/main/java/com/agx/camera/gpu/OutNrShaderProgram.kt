package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Stage 5-L - post-demosaic luma denoise: weighted YC1C2 + SWGF
 * self-guided filter (I=p, 8 side windows, min-variance window chosen) with a
 * separable box-statistic prefilter. The pass runs twice with a
 * beta=0.3 noise return; round 2 raises the epsilon multiplier to 1.96 = (kappa x 1.4)^2 -
 * "kappa x 1.4 on round 2" scales sigma, and epsilon proportional to sigma_hat^2 so the variance-domain
 * multiplier is the square.  Host side (PreviewRenderer.runStage5) runs both
 * iterations; slider 0 skips the pass entirely (bit-exact bypass).
 *
 * This program filters luma only. Chroma is passed through bit-exactly; the
 * chroma branch lives in S5-C (S5CShaderProgram) and reads this pass's luma as
 * its guide. The S5-L slider is a final linear blend between the
 * (beta-composed) input and this filtered result:
 * `mix(rgbIn, outRgb, u_strength)`, so 0..100 maps to 0%..100% of the filter.
 *
 * Passes (statsH/statsV/main run one texel per output pixel, same resolution as
 * the demosaic FBO):
 *  statsH  : horizontal 5-tap box of Y (and Y^2) of the (possibly beta-blended)
 *            iteration input.
 *  statsV  : vertical 5-tap box of the statsH output -> full 5x5 box stats.
 *  main    : YC1C2 -> per-pixel epsilon = scale*(sigma_hat^2 + sigma_dm^2), pick the min-variance side
 *            window (sampling statsV at 8 window centres), filter Y, keep chroma,
 *            and invert back to RGB. The support-shrinkage gate falls back to a
 *            3x3 mean wherever the centre window is cut by structure.
 *
 * The guided coefficient grid is not used: its 8-px upsample drew directional
 * streaks along a curved bright edge. The per-pixel window blend above is the
 * certified two-window estimate and is the shipped form.
 *  grainDcH / grainDc / grainStat / grainSmooth / grainApply
 *          : the film-grain return, run once after the last main pass and added
 *            to its output: the residual's DC as a horizontal pixel sum followed
 *            by a vertical cell sum, a strided gradient sample per cell, the
 *            frame gradient percentile from that plane in one 1x1 fragment, the
 *            smoothed residual, then the notch, the partition gains, the edge
 *            term and the dose. Sigma and the DC box are keyed to the frame
 *            height (grainLawFor), so the operator's units follow the render
 *            size; the box turns over at small heights and the kernel underflows
 *            to the identity there, both as grainLawFor describes.
 */
class OutNrShaderProgram {

    private var statsHProgramId = 0
    private var statsVProgramId = 0
    private var mainProgramId = 0
    private var grainDchProgramId = 0
    private var grainDcProgramId = 0
    private var grainStatProgramId = 0
    private var grainSmoothProgramId = 0
    private var grainApplyProgramId = 0
    private var ready = false

    /**
     * The main filter program only. The statistics programs are captured once
     * per frame into a 1D reduction rather than run per pixel, and the budget
     * counts per-texel work, so charging their size against this pass would
     * overstate it.
     */
    fun budgetProgramId(): Int = mainProgramId

    private var hInTexLoc = 0
    private var hBaseTexLoc = 0
    private var hBetaLoc = 0
    private var hYWeightsLoc = 0
    private var hWinScaleLoc = 0
    private var hFlagTexLoc = 0
    private var hFlagAvailLoc = 0

    private var vStatsTexLoc = 0
    private var vWinScaleLoc = 0
    private var vFlagAvailLoc = 0

    private var mInTexLoc = 0
    private var mBaseTexLoc = 0
    private var mStatsTexLoc = 0
    private var mSigmaTexLoc = 0
    private var mSigmaSizeLoc = 0
    private var mOutSizeLoc = 0
    private var mBetaLoc = 0
    private var mLumaEpsScaleLoc = 0
    private var mSigmaDm2Loc = 0
    private var mInverseRange2Loc = 0
    private var mEpsScaleLoc = 0
    private var mUseIsoSigmaLoc = 0
    private var mIsoModelALoc = 0
    private var mIsoModelBLoc = 0
    private var mWinScaleLoc = 0
    private var mEpsBoostLoc = 0
    private var mStrengthLoc = 0
    private var mShrinkLoc = 0
    private var mShrinkRefLoc = 0
    private var mShrinkTauLoc = 0
    private var mShrinkCapLoc = 0
    private var mFlagTexLoc = 0
    private var mFlagAvailLoc = 0

    private var gsInTexLoc = 0
    private var gsK0Loc = 0
    private var gsK1Loc = 0
    private var gsHatTexLoc = 0
    private var gsDcTexLoc = 0
    private var gsStatTexLoc = 0
    private var gsDcSizeLoc = 0
    private var gsOutSizeLoc = 0

    private var ghInTexLoc = 0
    private var ghHatTexLoc = 0
    private var ghInSizeLoc = 0
    private var ghHalfLoc = 0
    private var ghCellLoc = 0

    private var gdHTexLoc = 0
    private var gdInTexLoc = 0
    private var gdHSizeLoc = 0
    private var gdInSizeLoc = 0
    private var gdHalfLoc = 0
    private var gdVHalfLoc = 0
    private var gdCellLoc = 0

    private var gtGradTexLoc = 0
    private var gtPlaneSizeLoc = 0

    private var gaHatTexLoc = 0
    private var gaSTexLoc = 0
    private var gaGradRefTexLoc = 0
    private var gaOutSizeLoc = 0
    private var gaDoseLoc = 0
    private var gaFieldDoseLoc = 0

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        statsHProgramId = createProgram(VERTEX_SHADER, STATS_H_FRAGMENT)
        statsVProgramId = createProgram(VERTEX_SHADER, STATS_V_FRAGMENT)
        mainProgramId = createProgram(VERTEX_SHADER, MAIN_FRAGMENT)
        grainDchProgramId = createProgram(VERTEX_SHADER, GRAIN_DCH_FRAGMENT)
        grainDcProgramId = createProgram(VERTEX_SHADER, GRAIN_DC_FRAGMENT)
        grainStatProgramId = createProgram(VERTEX_SHADER, GRAIN_STAT_FRAGMENT)
        grainSmoothProgramId = createProgram(VERTEX_SHADER, GRAIN_SMOOTH_FRAGMENT)
        grainApplyProgramId = createProgram(VERTEX_SHADER, GRAIN_APPLY_FRAGMENT)
        if (statsHProgramId == 0 || statsVProgramId == 0 || mainProgramId == 0) {
            Log.e(TAG, "Failed to create Stage-5 denoise programs")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create Stage-5 denoise programs")
            return
        }

        hInTexLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_inTex")
        hBaseTexLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_baseTex")
        hBetaLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_beta")
        hYWeightsLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_yWeights")
        hWinScaleLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_win_scale")
        hFlagTexLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_flagTex")
        hFlagAvailLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_flag_available")

        vStatsTexLoc = GLES20.glGetUniformLocation(statsVProgramId, "u_statsTex")
        vWinScaleLoc = GLES20.glGetUniformLocation(statsVProgramId, "u_win_scale")
        vFlagAvailLoc = GLES20.glGetUniformLocation(statsVProgramId, "u_flag_available")

        gsInTexLoc = GLES20.glGetUniformLocation(grainSmoothProgramId, "u_inTex")
        gsHatTexLoc = GLES20.glGetUniformLocation(grainSmoothProgramId, "u_hatTex")
        gsDcTexLoc = GLES20.glGetUniformLocation(grainSmoothProgramId, "u_dcTex")
        gsStatTexLoc = GLES20.glGetUniformLocation(grainSmoothProgramId, "u_statTex")
        gsK0Loc = GLES20.glGetUniformLocation(grainSmoothProgramId, "u_k0")
        gsK1Loc = GLES20.glGetUniformLocation(grainSmoothProgramId, "u_k1")
        gsDcSizeLoc = GLES20.glGetUniformLocation(grainSmoothProgramId, "u_dcSize")
        gsOutSizeLoc = GLES20.glGetUniformLocation(grainSmoothProgramId, "u_outSize")

        ghInTexLoc = GLES20.glGetUniformLocation(grainDchProgramId, "u_inTex")
        ghHatTexLoc = GLES20.glGetUniformLocation(grainDchProgramId, "u_hatTex")
        ghInSizeLoc = GLES20.glGetUniformLocation(grainDchProgramId, "u_inSize")
        ghHalfLoc = GLES20.glGetUniformLocation(grainDchProgramId, "u_half")
        ghCellLoc = GLES20.glGetUniformLocation(grainDchProgramId, "u_cell")

        gdHTexLoc = GLES20.glGetUniformLocation(grainDcProgramId, "u_hTex")
        gdInTexLoc = GLES20.glGetUniformLocation(grainDcProgramId, "u_inTex")
        gdHSizeLoc = GLES20.glGetUniformLocation(grainDcProgramId, "u_hSize")
        gdInSizeLoc = GLES20.glGetUniformLocation(grainDcProgramId, "u_inSize")
        gdHalfLoc = GLES20.glGetUniformLocation(grainDcProgramId, "u_half")
        gdVHalfLoc = GLES20.glGetUniformLocation(grainDcProgramId, "u_vhalf")
        gdCellLoc = GLES20.glGetUniformLocation(grainDcProgramId, "u_cell")

        gtGradTexLoc = GLES20.glGetUniformLocation(grainStatProgramId, "u_gradTex")
        gtPlaneSizeLoc = GLES20.glGetUniformLocation(grainStatProgramId, "u_planeSize")

        gaHatTexLoc = GLES20.glGetUniformLocation(grainApplyProgramId, "u_hatTex")
        gaSTexLoc = GLES20.glGetUniformLocation(grainApplyProgramId, "u_sTex")
        gaGradRefTexLoc = GLES20.glGetUniformLocation(grainApplyProgramId, "u_gradRefTex")
        gaOutSizeLoc = GLES20.glGetUniformLocation(grainApplyProgramId, "u_outSize")
        gaDoseLoc = GLES20.glGetUniformLocation(grainApplyProgramId, "u_dose")
        gaFieldDoseLoc = GLES20.glGetUniformLocation(grainApplyProgramId, "u_fieldDose")

        mInTexLoc = GLES20.glGetUniformLocation(mainProgramId, "u_inTex")
        mBaseTexLoc = GLES20.glGetUniformLocation(mainProgramId, "u_baseTex")
        mStatsTexLoc = GLES20.glGetUniformLocation(mainProgramId, "u_statsTex")
        mSigmaTexLoc = GLES20.glGetUniformLocation(mainProgramId, "u_sigmaTex")
        mSigmaSizeLoc = GLES20.glGetUniformLocation(mainProgramId, "u_sigmaSize")
        mOutSizeLoc = GLES20.glGetUniformLocation(mainProgramId, "u_outSize")
        mBetaLoc = GLES20.glGetUniformLocation(mainProgramId, "u_beta")
        mLumaEpsScaleLoc = GLES20.glGetUniformLocation(mainProgramId, "u_lumaEpsScale")
        mSigmaDm2Loc = GLES20.glGetUniformLocation(mainProgramId, "u_sigma_dm2")
        mInverseRange2Loc = GLES20.glGetUniformLocation(mainProgramId, "u_inverse_range2")
        mEpsScaleLoc = GLES20.glGetUniformLocation(mainProgramId, "u_epsScale")
        mUseIsoSigmaLoc = GLES20.glGetUniformLocation(mainProgramId, "u_use_iso_sigma")
        mIsoModelALoc = GLES20.glGetUniformLocation(mainProgramId, "u_iso_model_a")
        mIsoModelBLoc = GLES20.glGetUniformLocation(mainProgramId, "u_iso_model_b")
        mWinScaleLoc = GLES20.glGetUniformLocation(mainProgramId, "u_win_scale")
        mEpsBoostLoc = GLES20.glGetUniformLocation(mainProgramId, "u_eps_boost")
        mStrengthLoc = GLES20.glGetUniformLocation(mainProgramId, "u_strength")
        mShrinkLoc = GLES20.glGetUniformLocation(mainProgramId, "u_shrink")
        mShrinkRefLoc = GLES20.glGetUniformLocation(mainProgramId, "u_shrink_ref")
        mShrinkTauLoc = GLES20.glGetUniformLocation(mainProgramId, "u_shrink_tau")
        mShrinkCapLoc = GLES20.glGetUniformLocation(mainProgramId, "u_shrink_cap")
        mFlagTexLoc = GLES20.glGetUniformLocation(mainProgramId, "u_flagTex")
        mFlagAvailLoc = GLES20.glGetUniformLocation(mainProgramId, "u_flag_available")

        ready = true
        Log.d(TAG, "Stage-5 denoise programs created: h=$statsHProgramId v=$statsVProgramId " +
            "m=$mainProgramId")
        com.agx.camera.CrashLogger.log(
            TAG,
            "Program created: outNr=($statsHProgramId,$statsVProgramId,$mainProgramId)"
        )
    }

    private fun drawQuad(programId: Int) {
        val posHandle = GLES20.glGetAttribLocation(programId, "a_position")
        val texHandle = GLES20.glGetAttribLocation(programId, "a_texCoord")
        if (posHandle >= 0) {
            GLES20.glEnableVertexAttribArray(posHandle)
            GLES20.glVertexAttribPointer(posHandle, 2, GLES20.GL_FLOAT, false, 0, quadVertices)
        }
        if (texHandle >= 0) {
            GLES20.glEnableVertexAttribArray(texHandle)
            GLES20.glVertexAttribPointer(texHandle, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
        }
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        if (posHandle >= 0) GLES20.glDisableVertexAttribArray(posHandle)
        if (texHandle >= 0) GLES20.glDisableVertexAttribArray(texHandle)
    }

    fun drawStatsH(
        inTex: Int, baseTex: Int, beta: Float, yWeights: FloatArray, winScale: Float = 1f,
        flagTex: Int = 0, flagAvailable: Boolean = false
    ) {
        if (statsHProgramId == 0) return
        GLES20.glUseProgram(statsHProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inTex)
        GLES20.glUniform1i(hInTexLoc, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, baseTex)
        GLES20.glUniform1i(hBaseTexLoc, 1)
        GLES20.glUniform1f(hBetaLoc, beta)
        GLES20.glUniform3f(hYWeightsLoc, yWeights[0], yWeights[1], yWeights[2])
        GLES20.glUniform1f(hWinScaleLoc, winScale)
        // Defect mask sampler: always bind a complete float texture (bind inTex
        // as the placeholder when the mask is unavailable) so the used sampler
        // never leaves a unit incomplete; the u_flag_available gate decides
        // whether the shader actually reads it.
        val ft = if (flagTex != 0) flagTex else inTex
        GLES20.glActiveTexture(GLES20.GL_TEXTURE4)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, ft)
        GLES20.glUniform1i(hFlagTexLoc, 4)
        GLES20.glUniform1f(hFlagAvailLoc, if (flagAvailable && flagTex != 0) 1f else 0f)
        drawQuad(statsHProgramId)
    }

    fun drawStatsV(statsTex: Int, winScale: Float = 1f, flagAvailable: Boolean = false) {
        if (statsVProgramId == 0) return
        GLES20.glUseProgram(statsVProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, statsTex)
        GLES20.glUniform1i(vStatsTexLoc, 0)
        GLES20.glUniform1f(vWinScaleLoc, winScale)
        GLES20.glUniform1f(vFlagAvailLoc, if (flagAvailable) 1f else 0f)
        drawQuad(statsVProgramId)
    }

    fun drawMain(
        inTex: Int, baseTex: Int, statsTex: Int, sigmaTex: Int,
        sigmaW: Float, sigmaH: Float, outW: Float, outH: Float,
        beta: Float, lumaEpsScale: Float, sigmaDm2: Float,
        inverseRange2: Float, epsScale: Float,
        useIsoSigma: Boolean = false,
        isoModelA: Float = 0f, isoModelB: Float = 0f,
        winScale: Float = 1f, epsBoost: Float = 1f,
        strength: Float = 1f,
        shrink: Boolean = true, shrinkRef: Float = SHRINK_REF, shrinkTau: Float = SHRINK_TAU,
        shrinkCap: Float = SHRINK_CAP,
        flagTex: Int = 0, flagAvailable: Boolean = false
    ) {
        if (mainProgramId == 0) return
        GLES20.glUseProgram(mainProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inTex)
        GLES20.glUniform1i(mInTexLoc, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, baseTex)
        GLES20.glUniform1i(mBaseTexLoc, 1)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, statsTex)
        GLES20.glUniform1i(mStatsTexLoc, 2)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE3)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sigmaTex)
        GLES20.glUniform1i(mSigmaTexLoc, 3)
        GLES20.glUniform2f(mSigmaSizeLoc, sigmaW, sigmaH)
        GLES20.glUniform2f(mOutSizeLoc, outW, outH)
        GLES20.glUniform1f(mBetaLoc, beta)
        GLES20.glUniform1f(mLumaEpsScaleLoc, lumaEpsScale)
        GLES20.glUniform1f(mSigmaDm2Loc, sigmaDm2)
        GLES20.glUniform1f(mInverseRange2Loc, inverseRange2)
        GLES20.glUniform1f(mEpsScaleLoc, epsScale)
        GLES20.glUniform1f(mUseIsoSigmaLoc, if (useIsoSigma) 1f else 0f)
        GLES20.glUniform1f(mIsoModelALoc, isoModelA)
        GLES20.glUniform1f(mIsoModelBLoc, isoModelB)
        GLES20.glUniform1f(mWinScaleLoc, winScale)
        GLES20.glUniform1f(mEpsBoostLoc, epsBoost)
        GLES20.glUniform1f(mStrengthLoc, strength)
        GLES20.glUniform1f(mShrinkLoc, if (shrink) 1f else 0f)
        GLES20.glUniform1f(mShrinkRefLoc, shrinkRef)
        GLES20.glUniform1f(mShrinkTauLoc, shrinkTau)
        GLES20.glUniform1f(mShrinkCapLoc, shrinkCap)
        val ft = if (flagTex != 0) flagTex else inTex
        GLES20.glActiveTexture(GLES20.GL_TEXTURE4)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, ft)
        GLES20.glUniform1i(mFlagTexLoc, 4)
        GLES20.glUniform1f(mFlagAvailLoc, if (flagAvailable && flagTex != 0) 1f else 0f)
        drawQuad(mainProgramId)
    }


    /**
     * S5-L grain gate, pass 0: the residual's DC on the 8-pixel grid, written at
     * 1/8 resolution. One texel per 8x8 block, each a plain mean of the block.
     *
     * The gate needs r minus its local mean so only fine content is returned;
     * the prototype takes a 31-px box, which no per-pixel pass can afford (it is
     * ~a quarter of S5-L's fetches on its own). Averaging on a coarse grid
     * carries the same window at 1/64 of the pixels - about two fetches per
     * output pixel.
     */
    fun drawGrainDcH(inTex: Int, hatTex: Int, outW: Int, outH: Int, half: Float, cell: Float) {
        if (grainDchProgramId == 0) return
        GLES20.glUseProgram(grainDchProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inTex)
        GLES20.glUniform1i(ghInTexLoc, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, hatTex)
        GLES20.glUniform1i(ghHatTexLoc, 1)
        GLES20.glUniform2i(ghInSizeLoc, outW, outH)
        GLES20.glUniform1f(ghHalfLoc, half)
        GLES20.glUniform1f(ghCellLoc, cell)
        drawQuad(grainDchProgramId)
    }

    /** Second half of the DC box: the vertical 31-tap sum, plus the gradient
     *  sample the frame percentile is taken over. Writes .r = DC, .g = |grad|. */
    fun drawGrainDc(
        hTex: Int, inTex: Int, hW: Float, hH: Float, outW: Int, outH: Int,
        half: Float, vhalf: Float, cell: Float
    ) {
        if (grainDcProgramId == 0) return
        GLES20.glUseProgram(grainDcProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, hTex)
        GLES20.glUniform1i(gdHTexLoc, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inTex)
        GLES20.glUniform1i(gdInTexLoc, 1)
        GLES20.glUniform2f(gdHSizeLoc, hW, hH)
        GLES20.glUniform2i(gdInSizeLoc, outW, outH)
        GLES20.glUniform1f(gdHalfLoc, half)
        GLES20.glUniform1f(gdVHalfLoc, vhalf)
        GLES20.glUniform1f(gdCellLoc, cell)
        drawQuad(grainDcProgramId)
    }

    /**
     * S5-L grain gate, pass 1: the frame gradient percentile, one fragment.
     * Reads the 1/8-resolution plane pass 0 wrote and returns the 90th
     * percentile of its .g in a 1x1 texture, which the apply pass reads.
     */
    fun drawGrainStat(gradTex: Int, planeW: Float, planeH: Float) {
        if (grainStatProgramId == 0) return
        GLES20.glUseProgram(grainStatProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, gradTex)
        GLES20.glUniform1i(gtGradTexLoc, 0)
        GLES20.glUniform2f(gtPlaneSizeLoc, planeW, planeH)
        drawQuad(grainStatProgramId)
    }

    /**
     * S5-L grain gate, pass 2: the smoothed, DC-free residual and the gate
     * input's gradient, at full resolution.
     */
    fun drawGrainSmooth(
        inTex: Int, hatTex: Int, dcTex: Int, statTex: Int,
        dcW: Float, dcH: Float,
        outW: Float, outH: Float, k0: Float, k1: Float
    ) {
        if (grainSmoothProgramId == 0) return
        GLES20.glUseProgram(grainSmoothProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inTex)
        GLES20.glUniform1i(gsInTexLoc, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, hatTex)
        GLES20.glUniform1i(gsHatTexLoc, 1)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, dcTex)
        GLES20.glUniform1i(gsDcTexLoc, 2)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE3)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, statTex)
        GLES20.glUniform1i(gsStatTexLoc, 3)
        GLES20.glUniform2f(gsDcSizeLoc, dcW, dcH)
        GLES20.glUniform2f(gsOutSizeLoc, outW, outH)
        GLES20.glUniform1f(gsK0Loc, k0)
        GLES20.glUniform1f(gsK1Loc, k1)
        drawQuad(grainSmoothProgramId)
    }

    /**
     * S5-L grain gate, pass 3: the lattice notch on the smoothed plane, the D4
     * band gains, the frame-p90 edge term and the dose; added to the S5-L
     * output.
     *
     * The likeness is 1 in this form: its reference is a frame median of the
     * local in-band share, which is a second reduction, and substituting 1 into
     * the prototype moves its reference rows by ~4%.
     */
    fun drawGrainApply(
        hatTex: Int, sTex: Int, gradRefTex: Int, outW: Float, outH: Float,
        dose: Float, fieldDose: Float
    ) {
        if (grainApplyProgramId == 0) return
        GLES20.glUseProgram(grainApplyProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, hatTex)
        GLES20.glUniform1i(gaHatTexLoc, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sTex)
        GLES20.glUniform1i(gaSTexLoc, 1)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, gradRefTex)
        GLES20.glUniform1i(gaGradRefTexLoc, 2)
        GLES20.glUniform2f(gaOutSizeLoc, outW, outH)
        GLES20.glUniform1f(gaDoseLoc, dose)
        GLES20.glUniform1f(gaFieldDoseLoc, fieldDose)
        drawQuad(grainApplyProgramId)
    }

    /**
     * The gate's two resolution-dependent numbers, from the frame height.
     *
     * The prototype defines the grain sigma as a fraction of the frame (6.5 um on
     * a 24 mm negative, 0.378 px at 3072 px of height), so the sigma scales with
     * the height the stage runs at. Its zero-DC box follows the same intent by
     * extrapolation - LOCAL_WIN 15 is a fixed pixel count in the prototype, so the
     * prototype's own renders cannot check the scaling. At 3072 this returns
     * exactly the calibrated pair; as the height falls the two side taps of the
     * kernel underflow and it becomes the identity, which is the right answer for
     * grain under a pixel wide.
     */
    fun grainLawFor(outH: Int): FloatArray {
        val h = outH.coerceAtLeast(1).toFloat()
        // 0.454 x (6.5 um / 24 mm) x height: the prototype's dye-cloud sigma as a
        // fraction of the frame, 0.37779 px at 3072.
        val law = 1.22979e-4f * h
        // Below the calibrated height the law puts the dye cloud under a pixel,
        // where no kernel can band-pass it and the filter returns the residual's
        // structure instead - the blotches beside a hard edge. There the grain is
        // placed at the finest scale a pixel grid can carry. The calibrated height
        // keeps the law exactly: the floor is above it and must never reach it.
        val lowHeight = h < GRAIN_LOW_HEIGHT
        val sigma = if (lowHeight) maxOf(law, GRAIN_SIGMA_FLOOR) else law
        val k1 = Math.exp(-0.5 / (sigma * sigma).toDouble()).toFloat()
        val k2 = Math.exp(-2.0 / (sigma * sigma).toDouble()).toFloat()
        // The prototype's normalisation: five sampled taps, of which the outer
        // pair is dropped after normalising (8e-7 at this sigma).
        val norm = 1f / (1f + 2f * k1 + 2f * k2)
        val k0 = norm
        val half = Math.round(15f * h / 3072f).toFloat().coerceIn(1f, 15f)
        return floatArrayOf(k0, k1 * norm, half)
    }

    /** Zero-DC cell in pixels: the 8-px grid is the calibrated form, and below
     *  the calibrated height it cannot hold the box (11 px across needs 8-px
     *  cells that then span 24 rows - a 2.2:1 anisotropy the return turns into
     *  blotches), so the plane goes to full resolution and the box is exact. */
    fun grainCellFor(outH: Int): Int = if (outH < GRAIN_LOW_HEIGHT) 1 else 8

    /** The grain gate is part of S5-L; when it fails to compile the stage still
     *  runs, just without it. */
    fun grainReady(): Boolean = grainDchProgramId != 0 && grainDcProgramId != 0 &&
        grainStatProgramId != 0 &&
        grainSmoothProgramId != 0 && grainApplyProgramId != 0

    fun isReady(): Boolean = ready && statsHProgramId != 0 && statsVProgramId != 0 &&
        mainProgramId != 0

    fun destroy() {
        GLES20.glDeleteProgram(statsHProgramId)
        GLES20.glDeleteProgram(statsVProgramId)
        GLES20.glDeleteProgram(mainProgramId)
        GLES20.glDeleteProgram(grainDchProgramId)
        GLES20.glDeleteProgram(grainDcProgramId)
        GLES20.glDeleteProgram(grainStatProgramId)
        GLES20.glDeleteProgram(grainSmoothProgramId)
        GLES20.glDeleteProgram(grainApplyProgramId)
        ready = false
        grainDchProgramId = 0
        grainDcProgramId = 0
        grainStatProgramId = 0
        grainSmoothProgramId = 0
        grainApplyProgramId = 0
        statsHProgramId = 0
        statsVProgramId = 0
        mainProgramId = 0
    }

    companion object {
        private const val TAG = "OutNrShaderProgram"

        private val QUAD_COORDS = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        private val QUAD_TEX_COORDS = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f)

        /**
         * Support-shrinkage calibration. The flat-patch window variance sits
         * near 0.105 epsilon on the reference chain (measured against the
         * mirror in pysim, ISO 1600 and 3200 frames agreeing to 10%), and the
         * gate opens fully once the excess reaches one tau.
         */
        private const val SHRINK_REF = 0.105f
        private const val SHRINK_TAU = 2.0f
        private const val SHRINK_CAP = 0.5f
        // Below this height the grain law's dye cloud is under a pixel and the
        // zero-DC box no longer fits the 8-px cell grid; the gate switches to the
        // resolvable form there. The calibrated 3072 form is above it and keeps
        // the law untouched. This is a hard cliff, not a taper: 2047 rows takes
        // the 0.7-px floor and cell 1, 2048 rows takes the law and cell 8.
        private const val GRAIN_LOW_HEIGHT = 2048f
        // Finest grain a pixel grid can carry as a band-pass. The 1080 and 720
        // measurements put the bead row out only at 0.7 (0.5 left +0.5 of it).
        private const val GRAIN_SIGMA_FLOOR = 0.7f

        /** 8 SWGF side-window centres (in output pixels), r = 2. */
        val WINDOW_CENTERS = arrayOf(
            intArrayOf(-2, 0), intArrayOf(2, 0), intArrayOf(0, -2), intArrayOf(0, 2),
            intArrayOf(-2, -2), intArrayOf(2, -2), intArrayOf(-2, 2), intArrayOf(2, 2)
        )

        private const val VERTEX_SHADER = """
#version 300 es
in vec2 a_position;
in vec2 a_texCoord;
out vec2 v_texCoord;
uniform mat4 u_transformMatrix;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = (u_transformMatrix * vec4(a_texCoord, 0.0, 1.0)).xy;
}
"""

        private const val STATS_H_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_inTex;
uniform sampler2D u_baseTex;
uniform sampler2D u_flagTex;
uniform float u_beta;
uniform vec3 u_yWeights;
// 1.0 => the defect mask is valid: flagged taps are excluded from the box
// statistics and the surviving count is carried in .b (see header note).
uniform float u_flag_available;
// Scaling the stats box alongside the window centres keeps the 8 SWGF
// candidate windows self-similar at any resolution: preview (winScale=1)
// keeps the exact 5-tap box (R=2, bit-identical), while the capture path
// (winScale>1) widens H and V identically so the boxes tile the plane as
// densely as the preview.  Without this the capture windows drift apart and
// the min-variance/distance selection aliases into blocky patches.
uniform float u_win_scale;

float lumaOf(vec3 rgb) { return dot(u_yWeights, rgb); }

// Defect mask convention: a texel is flagged if ANY channel magnitude exceeds
// 0.5.  The preview's DPC flag map stores +-1 in the owning phase channel(s)
// (hot/cold), the capture map writes 0/1 to all four channels - one test
// covers both, with no second indexing scheme.
float flagOf(ivec2 t) {
    vec4 f = texelFetch(u_flagTex, t, 0);
    return (abs(f.r) + abs(f.g) + abs(f.b) + abs(f.a) > 0.5) ? 1.0 : 0.0;
}

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);
    int R = max(1, int(round(2.0 * u_win_scale)));
    float sum = 0.0;
    float sumSq = 0.0;
    if (u_flag_available > 0.5) {
        // Defect-aware GF box: flagged taps are dropped and the mean/variance
        // are renormalized by the surviving count, written to .b so statsV can
        // combine the two axes with the exact per-tap weight.  When a whole row
        // is flagged the row is left empty (count 0) and statsV falls back.
        float cnt = 0.0;
        for (int dx = -R; dx <= R; dx++) {
            ivec2 t = base + ivec2(dx, 0);
            if (flagOf(t) > 0.5) continue;
            vec3 a = texelFetch(u_inTex, t, 0).rgb;
            vec3 b = texelFetch(u_baseTex, t, 0).rgb;
            vec3 rgb = mix(a, b, u_beta);
            float y = lumaOf(rgb);
            sum += y;
            sumSq += y * y;
            cnt += 1.0;
        }
        if (cnt < 0.5) outColor = vec4(0.0, 0.0, 0.0, 0.0);
        else outColor = vec4(sum / cnt, sumSq / cnt, cnt, 0.0);
    } else {
        for (int dx = -R; dx <= R; dx++) {
            ivec2 t = base + ivec2(dx, 0);
            vec3 a = texelFetch(u_inTex, t, 0).rgb;
            vec3 b = texelFetch(u_baseTex, t, 0).rgb;
            vec3 rgb = mix(a, b, u_beta);
            float y = lumaOf(rgb);
            sum += y;
            sumSq += y * y;
        }
        outColor = vec4(sum / float(2 * R + 1), sumSq / float(2 * R + 1), 0.0, 0.0);
    }
}
"""

        private const val STATS_V_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_statsTex;
uniform float u_win_scale;
uniform float u_flag_available;

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);
    int R = max(1, int(round(2.0 * u_win_scale)));
    float sum = 0.0;
    float sumSq = 0.0;
    if (u_flag_available > 0.5) {
        // statsH rows carry their surviving-sample count in .b; weight the
        // (already renormalized) row means by it to reconstruct the true 2D box
        // mean/variance with flagged taps removed.  An all-flagged row has
        // count 0 and contributes nothing; count is forwarded for statsAt().
        float cnt = 0.0;
        for (int dy = -R; dy <= R; dy++) {
            vec3 s = texelFetch(u_statsTex, base + ivec2(0, dy), 0).rgb;
            sum += s.r * s.b;
            sumSq += s.g * s.b;
            cnt += s.b;
        }
        if (cnt < 0.5) outColor = vec4(0.0, 0.0, 0.0, 0.0);
        else outColor = vec4(sum / cnt, sumSq / cnt, cnt, 0.0);
    } else {
        for (int dy = -R; dy <= R; dy++) {
            vec2 s = texelFetch(u_statsTex, base + ivec2(0, dy), 0).rg;
            sum += s.r;
            sumSq += s.g;
        }
        outColor = vec4(sum / float(2 * R + 1), sumSq / float(2 * R + 1), 0.0, 0.0);
    }
}
"""

        /**
         * S5-L grain gate, pass 0: the horizontal 31-tap sum of the residual, on
         * the 8-pixel grid.
         *
         * The window is the prototype's (LOCAL_WIN 15 -> a 31-px box), not a
         * block mean: a mean over the notch's own 2x2 cell cancels the very
         * block means the projection keeps, and an 8-pixel block mean did
         * exactly that, returning 70% of the prototype's band energy. Split into
         * two axes at 1/8 resolution the box costs about one fetch per output
         * pixel.
         */
        private const val GRAIN_DCH_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform sampler2D u_inTex;
uniform sampler2D u_hatTex;
uniform ivec2 u_inSize;
uniform float u_half;
uniform float u_cell;

const vec3 YW = vec3(0.25, 0.5, 0.25);

void main() {
    int cellPx = max(1, int(u_cell + 0.5));
    ivec2 c = ivec2(gl_FragCoord.xy) * cellPx + ivec2(cellPx / 2);
    ivec2 hi = u_inSize - ivec2(1);
    float acc = 0.0;
    float acc2 = 0.0;
    for (int i = -15; i <= 15; i++) {
        if (abs(float(i)) <= u_half) {
            ivec2 p = clamp(c + ivec2(i, 0), ivec2(0), hi);
            float r = dot(texelFetch(u_inTex, p, 0).rgb, YW)
                    - dot(texelFetch(u_hatTex, p, 0).rgb, YW);
            acc += r;
            acc2 += r * r;
        }
    }
    outColor = vec4(acc, acc2, 0.0, 1.0);
}
"""

        /**
         * S5-L grain gate, pass 1: the vertical sum that completes the DC box -
         * counted in cells, since the horizontal plane is at 1/8 resolution - plus
         * one strided gradient sample of the gate input at the cell's centre (the
         * plane the frame percentile is taken over).
         */
        private const val GRAIN_DC_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform sampler2D u_hTex;
uniform sampler2D u_inTex;
uniform vec2 u_hSize;
uniform ivec2 u_inSize;
uniform float u_half;
uniform float u_vhalf;
uniform float u_cell;

const vec3 YW = vec3(0.25, 0.5, 0.25);

float lumaIn(ivec2 p) {
    return dot(texelFetch(u_inTex, clamp(p, ivec2(0), u_inSize - ivec2(1)), 0).rgb, YW);
}

void main() {
    int cellPx = max(1, int(u_cell + 0.5));
    ivec2 cell = ivec2(gl_FragCoord.xy);
    ivec2 lim = ivec2(u_hSize) - ivec2(1);
    // This plane is one texel per cell, so the vertical window is counted in
    // cells, not pixels: u_vhalf = round(u_half / cell). At 3072 the cell is 8 px
    // and the box is 31 px across by 5 cells (40 px) down - the closest square
    // that grid allows, and the reason the divisor is not 31x31. Below the
    // calibrated height the cell shrinks to 1 px, which makes the box the exact
    // square the prototype uses: at 1080 an 8-px cell can only give 11x24, and
    // that 2.2:1 anisotropy is what the return turns into blotches.
    float acc = 0.0;
    float acc2 = 0.0;
    for (int j = -15; j <= 15; j++) {
        if (abs(float(j)) <= u_vhalf) {
            vec4 h = texelFetch(u_hTex, clamp(cell + ivec2(0, j), ivec2(0), lim), 0);
            acc += h.r;
            acc2 += h.g;
        }
    }
    float taps = (2.0 * u_half + 1.0) * (2.0 * u_vhalf + 1.0);
    ivec2 c = cell * cellPx + ivec2(cellPx / 2);
    float gl = lumaIn(c - ivec2(1, 0));
    float gr = lumaIn(c + ivec2(1, 0));
    float gd = lumaIn(c - ivec2(0, 1));
    float gu = lumaIn(c + ivec2(0, 1));
    float gmag = 0.5 * sqrt((gr - gl) * (gr - gl) + (gu - gd) * (gu - gd));
    // .b is the residual's box variance: the detail weight's evidence, and the
    // only extra quantity this pass has to carry. It costs ALU, not a fetch.
    float mu = acc / taps;
    float mu2 = acc2 / taps;
    outColor = vec4(mu, gmag, max(mu2 - mu * mu, 0.0), 1.0);
}
"""

        /**
         * S5-L grain gate, pass 2: the smoothed, DC-free residual, plus the
         * gate input's own gradient in .g.
         *
         * The prototype smooths before it notches, and so does this: the notch
         * is a block difference of this plane, taken in the apply pass. The DC
         * is subtracted after the blur rather than before it - the DC is a
         * 31-px by 5-cell mean, so blurring it changes it by well under a percent,
         * and subtracting it once here costs four taps instead of thirty-six.
         */
        private const val GRAIN_SMOOTH_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform sampler2D u_inTex;
uniform sampler2D u_hatTex;
uniform sampler2D u_dcTex;
uniform sampler2D u_statTex;
uniform vec2 u_dcSize;
uniform vec2 u_outSize;
uniform float u_k0;
uniform float u_k1;

const vec3 YW = vec3(0.25, 0.5, 0.25);

float resid(vec2 uv) {
    return dot(texture(u_inTex, uv).rgb, YW) - dot(texture(u_hatTex, uv).rgb, YW);
}

float lumaIn(vec2 uv) { return dot(texture(u_inTex, uv).rgb, YW); }

float dcAt(vec2 px) {
    vec2 g = (px + vec2(0.5)) * (u_dcSize / u_outSize) - vec2(0.5);
    vec2 g0 = floor(g);
    vec2 f = g - g0;
    ivec2 hi = ivec2(u_dcSize) - ivec2(1);
    ivec2 i0 = ivec2(g0);
    float c00 = texelFetch(u_dcTex, clamp(i0, ivec2(0), hi), 0).r;
    float c10 = texelFetch(u_dcTex, clamp(i0 + ivec2(1, 0), ivec2(0), hi), 0).r;
    float c01 = texelFetch(u_dcTex, clamp(i0 + ivec2(0, 1), ivec2(0), hi), 0).r;
    float c11 = texelFetch(u_dcTex, clamp(i0 + ivec2(1, 1), ivec2(0), hi), 0).r;
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// The Gaussian at the film-grain sigma, on three taps of the sampled kernel.
// The weights come from the host, computed from the frame height (0.943335 /
// 0.028328 at 3072 px); the grain's sigma is a fraction of the frame, so at
// small heights the two side taps underflow and the kernel is the identity.
float smooth3(vec2 uv, vec2 px) {
    float k0 = u_k0;
    float k1 = u_k1;
    float cx = k0 * resid(uv) + k1 * (resid(uv + vec2(px.x, 0.0)) + resid(uv - vec2(px.x, 0.0)));
    float up = k0 * resid(uv + vec2(0.0, px.y)) + k1 * (resid(uv + px) + resid(uv + vec2(px.x, -px.y)));
    float dn = k0 * resid(uv - vec2(0.0, px.y)) + k1 * (resid(uv + vec2(-px.x, px.y)) + resid(uv - px));
    return k0 * cx + k1 * (up + dn);
}

// The same nine samples, averaged: this is the band's low cut. The coarse DC
// plane leaves the denoiser's halo around a bright edge in the return; at 3x3
// the halo is gone and the grain-scale content stays. No extra fetch, since
// smooth3 already reads all nine.
float box3(vec2 uv, vec2 px) {
    float c  = resid(uv);
    float l  = resid(uv - vec2(px.x, 0.0));
    float r  = resid(uv + vec2(px.x, 0.0));
    float tp = resid(uv + vec2(0.0, px.y));
    float bt = resid(uv - vec2(0.0, px.y));
    float tl = resid(uv + vec2(-px.x, px.y));
    float tr = resid(uv + px);
    float bl = resid(uv - px);
    float br = resid(uv + vec2(px.x, -px.y));
    return (c + l + r + tp + bt + tl + tr + bl + br) * (1.0 / 9.0);
}

// Bilinear read of the DC plane's variance channel: the residual's local box
// energy. Interpolated exactly like dcAt, so the detail weight varies smoothly
// and cannot draw a contour of its own.
float dcVarAt(vec2 px) {
    vec2 g = (px + vec2(0.5)) * (u_dcSize / u_outSize) - vec2(0.5);
    vec2 g0 = floor(g);
    vec2 f = g - g0;
    ivec2 hi = ivec2(u_dcSize) - ivec2(1);
    ivec2 i0 = ivec2(g0);
    float c00 = texelFetch(u_dcTex, clamp(i0, ivec2(0), hi), 0).b;
    float c10 = texelFetch(u_dcTex, clamp(i0 + ivec2(1, 0), ivec2(0), hi), 0).b;
    float c01 = texelFetch(u_dcTex, clamp(i0 + ivec2(0, 1), ivec2(0), hi), 0).b;
    float c11 = texelFetch(u_dcTex, clamp(i0 + ivec2(1, 1), ivec2(0), hi), 0).b;
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

void main() {
    vec2 px = 1.0 / u_outSize;
    vec2 uv = gl_FragCoord.xy * px;
    // The band's low cut is the 3x3 mean of the residual, not the coarse DC
    // plane: the wide box left the denoiser's halo around a bright edge in the
    // return, which showed as a pale band along a lit cable.
    float s = smooth3(uv, px) - box3(uv, px);
    float yl = lumaIn(uv - vec2(px.x, 0.0));
    float yr = lumaIn(uv + vec2(px.x, 0.0));
    float yd = lumaIn(uv - vec2(0.0, px.y));
    float yu = lumaIn(uv + vec2(0.0, px.y));
    float grad = 0.5 * sqrt((yr - yl) * (yr - yl) + (yu - yd) * (yu - yd));
    // Detail weight: the residual's box energy against the frame's flat-area
    // baseline, open above a 1.5x knee. .b carries it to the apply pass, which
    // blends the genuine band with the paper-shaped field on it.
    float var = dcVarAt(floor(gl_FragCoord.xy));
    float base = max(texelFetch(u_statTex, ivec2(0), 0).g, 1e-12);
    // The baseline is the frame's MEAN box variance, so var/base sits around 1
    // almost everywhere and (var/base - 1) * 2 pins to 1 across the frame --
    // which gates the paper field out of the capture entirely. Open the knee
    // above twice the mean so only structure claims the genuine branch.
    // The DC box is 31x40, far wider than the 11x11 band the prototype weights on,
    // so its variance rises slowly over print and P sat. low: legibility 0.783
    // against the prototype weight's 0.802 on the manual. A 4x gain on the weight
    // (2.0 -> 8.0 here) closes it: stroke 0.104, legibility 0.806, grain 0.030.
    float pDet = clamp((var / base - 1.0) * 8.0, 0.0, 1.0);
    outColor = vec4(s, grad, pDet, 1.0);
}
"""

        /**
         * S5-L grain gate, pass 3: the frame gradient percentile, one fragment.
         *
         * The apply pass needs the input gradient's 90th percentile for the whole
         * frame, which is a reduction: this samples the 1/8-resolution plane on a
         * 64x64 grid (4096 strided samples, the count the prototype's own
         * stride-8 sample reaches on a full frame), takes the mean to set the
         * threshold span, then counts the samples above 16 levels of it and
         * interpolates the crossing. One fragment per frame, ~8k texel fetches,
         * no readback.
         */
        private const val GRAIN_STAT_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform sampler2D u_gradTex;
uniform vec2 u_planeSize;

const int N = 64;
const int NB = 16;

float sample_at(ivec2 ij) {
    vec2 step = u_planeSize / float(N);
    ivec2 p = ivec2((vec2(ij) + vec2(0.5)) * step);
    return texelFetch(u_gradTex, clamp(p, ivec2(0), ivec2(u_planeSize) - ivec2(1)), 0).g;
}

// The same strided sample on the DC plane's variance channel. This is the
// frame's flat-area residual energy: the baseline the detail weight divides by.
float sample_var(ivec2 ij) {
    vec2 step = u_planeSize / float(N);
    ivec2 p = ivec2((vec2(ij) + vec2(0.5)) * step);
    return texelFetch(u_gradTex, clamp(p, ivec2(0), ivec2(u_planeSize) - ivec2(1)), 0).b;
}

void main() {
    float sum = 0.0;
    float sumv = 0.0;
    for (int j = 0; j < N; j += 2) {
        for (int i = 0; i < N; i += 2) {
            sum += sample_at(ivec2(i, j));
            sumv += sample_var(ivec2(i, j));
        }
    }
    // Stride-2 sampling: 1024 samples of a smooth plane, same quantiles.
    float n = float((N / 2) * (N / 2));
    float mu = sum / n;
    float lo = max(0.2 * mu, 1e-9);
    float hi = max(4.0 * mu, 2.0 * lo);
    float cnt[NB];
    for (int b = 0; b < NB; b++) cnt[b] = 0.0;
    for (int j = 0; j < N; j += 2) {
        for (int i = 0; i < N; i += 2) {
            float v = sample_at(ivec2(i, j));
            for (int b = 0; b < NB; b++) {
                float t = lo + (hi - lo) * float(b) / float(NB - 1);
                cnt[b] += (v > t) ? 1.0 : 0.0;
            }
        }
    }
    float want = 0.9 * n;
    float ref = hi;
    for (int b = 0; b < NB - 1; b++) {
        if (cnt[b] >= want && cnt[b + 1] < want) {
            float t0 = lo + (hi - lo) * float(b) / float(NB - 1);
            float t1 = lo + (hi - lo) * float(b + 1) / float(NB - 1);
            ref = mix(t0, t1, clamp((cnt[b] - want) / max(cnt[b] - cnt[b + 1], 1e-6), 0.0, 1.0));
        }
    }
    // The detail weight's baseline is the FLAT level of the box variance, not its
    // frame mean: on a real crop the mean sits ~70x above the flat level, so a
    // mean baseline opens the genuine branch only halfway over texture. The
    // median of the same strided samples is the frame-relative stand-in for the
    // prototype's flat-mask median, and it needs no constant.
    float mu2 = sumv / n;
    float lo2 = max(0.2 * mu2, 1e-30);
    float hi2 = max(8.0 * mu2, 2.0 * lo2);
    float cnt2[NB];
    for (int b = 0; b < NB; b++) cnt2[b] = 0.0;
    // Stride 2 over the same grid: 1024 samples instead of 4096. The baseline is
    // a quantile of a smooth plane, so the coarser sample changes it by well under
    // a percent, and this loop was the gate's most expensive pass (3.0 ms against
    // the apply pass's 0.2 ms).
    float nv = 0.0;
    for (int j = 0; j < N; j += 2) {
        for (int i = 0; i < N; i += 2) {
            float v = sample_var(ivec2(i, j));
            nv += 1.0;
            for (int b = 0; b < NB; b++) {
                float t = lo2 + (hi2 - lo2) * float(b) / float(NB - 1);
                cnt2[b] += (v > t) ? 1.0 : 0.0;
            }
        }
    }
    float want2 = 0.5 * nv;
    float ref2 = hi2;
    for (int b = 0; b < NB - 1; b++) {
        if (cnt2[b] >= want2 && cnt2[b + 1] < want2) {
            float t0 = lo2 + (hi2 - lo2) * float(b) / float(NB - 1);
            float t1 = lo2 + (hi2 - lo2) * float(b + 1) / float(NB - 1);
            ref2 = mix(t0, t1, clamp((cnt2[b] - want2) / max(cnt2[b] - cnt2[b + 1], 1e-6), 0.0, 1.0));
        }
    }
    outColor = vec4(ref, ref2, 0.0, 1.0);
}
"""

        /**
         * S5-L grain gate, pass 4: the lattice notch on the smoothed residual,
         * the D4 partition gains, the frame-p90 edge term, the support-shrinkage
         * weight the main pass leaves in its alpha, and the dose; the return is
         * added to the S5-L output.
         *
         * The notch is the prototype's projection: per 2x2 block the three
         * Nyquist-mode amplitudes are (a-b-c+d)/4, (a-b+c-d)/4 and
         * (a+b-c-d)/4, and each pixel subtracts its own combination. It runs
         * here, on the smoothed plane, which is the prototype's order. The
         * likeness term is 1: its reference is a frame median of the local
         * in-band share, and substituting 1 into the prototype moves its
         * reference rows by ~4%.
         */
        private const val GRAIN_APPLY_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform sampler2D u_hatTex;
uniform sampler2D u_sTex;
uniform sampler2D u_gradRefTex;
uniform vec2 u_outSize;
uniform float u_dose;
uniform float u_fieldDose;

const vec3 YW = vec3(0.25, 0.5, 0.25);

// Hash noise for the grain field: no texture, no fetch, plain ALU.
float gnoise(ivec2 p) {
    vec2 q = vec2(p);
    q = fract(q * vec2(0.1031, 0.1030));
    q += dot(q, q.yx + 33.33);
    return fract((q.x + q.y) * q.x) - 0.5;
}

// The paper's two-diameter mixture at sensor scale: the 6.5 um cloud is
// sub-pixel (0.378 px at 3072) so it samples as white; the 15 um cloud is about
// one pixel and samples as a 3x3 mean of the SAME hash. The halves are
// correlated and not equal power: measured on 1024x1024 the 3x3 half carries
// 1/12 of the white half's power, the field std is 0.80 and its mean |x| is
// 0.686. The radial spectrum is 0.47 dB rms from W' (p90 0.66), with most of
// the deviation at the high-frequency end.
float fieldAt(ivec2 p) {
    float w = gnoise(p);
    float s = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            s += gnoise(p + ivec2(i, j));
        }
    }
    return 0.7071 * (w * 3.4641 + (s / 9.0) * 3.0);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_outSize;
    ivec2 ipo = ivec2(floor(gl_FragCoord.xy));
    ivec2 lim = ivec2(u_outSize) - ivec2(1);

    vec4 hat = texture(u_hatTex, uv);
    float yh = dot(hat.rgb, YW);
    vec4 c = texture(u_sTex, uv);
    float sC = c.r;

    // The projection is a difference of the block's four samples, so it is
    // invariant to what is common to them.
    ivec2 blk = (ipo / 2) * 2;
    float aa = texelFetch(u_sTex, clamp(blk, ivec2(0), lim), 0).r;
    float bb = texelFetch(u_sTex, clamp(blk + ivec2(1, 0), ivec2(0), lim), 0).r;
    float cc = texelFetch(u_sTex, clamp(blk + ivec2(0, 1), ivec2(0), lim), 0).r;
    float dd = texelFetch(u_sTex, clamp(blk + ivec2(1, 1), ivec2(0), lim), 0).r;
    float mxy = 0.25 * (aa - bb - cc + dd);
    float mx = 0.25 * (aa - bb + cc - dd);
    float my = 0.25 * (aa + bb - cc - dd);
    float sx = 1.0 - 2.0 * mod(float(ipo.x), 2.0);
    float sy = 1.0 - 2.0 * mod(float(ipo.y), 2.0);
    float band = sC - (sx * sy * mxy + sx * mx + sy * my);

    float bg = (yh < 0.0) ? 0.2 : ((yh < 0.08) ? 1.0 : 0.5);
    float ref = max(texelFetch(u_gradRefTex, ivec2(0), 0).r, 1e-12);
    float edgeW = clamp(1.0 - c.g / ref, 0.0, 1.0);
    // The weight in .a is the S5-L support-shrinkage coherence, already mapped
    // to 0..1 by the main pass; a whole window (or a path without the shrink
    // test) reads 1 and leaves the shipped return unchanged.
    //
    // .b is the detail weight: where the residual carries structure the genuine
    // band is returned, and where it does not the paper-shaped field replaces it
    // so the surfaces get film grain rather than the denoiser's leftover.
    float pDet = clamp(c.b, 0.0, 1.0);
    float field = fieldAt(ipo);
    // Detail keeps the genuine band; the rest of the surface gets the paper
    // field, which is what removes both the too-coarse leftover and the
    // too-fine residual, leaving the grain the paper specifies.
    // The field is synthetic: its amplitude is a fixed absolute dose, not
    // measured from the image. The D4 gains and the edge term are properties of
    // the RESIDUAL and must not scale it, or the port comes out ~3x quieter than
    // the operator the sim measured. The coherence still gates the whole return.
    float ret = u_dose * bg * edgeW * band * hat.a * pDet
              + u_fieldDose * field * hat.a * (1.0 - pDet);
    outColor = vec4(hat.rgb + vec3(ret), 1.0);
}
"""

        private const val MAIN_FRAGMENT = """
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_inTex;
uniform sampler2D u_baseTex;
uniform sampler2D u_statsTex;
uniform sampler2D u_sigmaTex;
uniform sampler2D u_flagTex;
uniform float u_flag_available;
uniform vec2 u_sigmaSize;
uniform vec2 u_outSize;
uniform float u_beta;
uniform float u_lumaEpsScale;
uniform float u_sigma_dm2;
uniform float u_inverse_range2;
uniform float u_epsScale;
// Resolution alignment: S5 geometry lives in output-pixel space, so at
// capture resolution (up to 3-4x the preview FBO) the fixed +-2 px windows
// cover only a 3-4x smaller image footprint than on the preview, weakening
// the low-frequency smoothing.  u_win_scale scales the 8 SWGF window centres
// so the box covers the same relative image region at any resolution.
// u_eps_boost is the capture-only epsilon multiplier: the preview
// sees the S3 + 4x4 boxAA-denoised residual (sigma_hat^2/32), but a 1:1 capture has
// no boxAA, so its per-pixel residual is ~boxAA^2 ~ 16x larger; boosting epsilon
// restores aY parity so the still smooths as hard as the preview does.
uniform float u_win_scale;
uniform float u_eps_boost;
uniform float u_strength;
// Support shrinkage (0 = off). The box mean is unbiased only while the window
// holds a single surface; where an edge, a flap or a clipped plateau cuts it,
// the mean is pulled across the boundary, and that bias is what shows as a
// halo beside a high-contrast edge and as a ragged fringe along a clipped
// region. u_shrink_ref is the flat-patch window variance in units of epsilon,
// a chain constant (the same kind of calibration as u_epsScale, not a device
// statistic); u_shrink_tau is how much excess opens the gate fully.
uniform float u_shrink;
uniform float u_shrink_ref;
uniform float u_shrink_tau;
// A ceiling on the coherence. Without it a faint marking survives only where
// the window variance rises, and a low-contrast label sits below that, so the
// wide window averages it away. Holding half of the small-window mean in every
// pixel keeps that kind of detail without giving up the boundary behaviour.
uniform float u_shrink_cap;
uniform float u_use_iso_sigma;
uniform float u_iso_model_a;
uniform float u_iso_model_b;

vec3 yccOf(vec3 rgb) {
    float y = 0.25 * rgb.r + 0.5  * rgb.g + 0.25 * rgb.b;
    float c1 = rgb.r - rgb.b;
    float c2 = 0.5 * (rgb.r + rgb.b) - rgb.g;
    return vec3(y, c1, c2);
}

vec3 rgbOf(vec3 ycc) {
    float y = ycc.x; float c1 = ycc.y; float c2 = ycc.z;
    return vec3(y + 0.5*c1 + 0.5*c2, y - 0.5*c2, y - 0.5*c1 + 0.5*c2);
}

// stats at a window centre in output-pixel coordinates:
// .xy = mean / mean-square, .z = surviving-sample count (0 when the mask is
// unavailable, the full box count otherwise).
vec3 statsAt(ivec2 base, ivec2 winCentre) {
    return texelFetch(u_statsTex, base + winCentre, 0).rgb;
}

// Defect mask convention: a texel is flagged if ANY channel magnitude exceeds
// 0.5 (preview +-1 per-phase DPC map and capture 0/1 map share this test).
float flagOf(ivec2 t) {
    vec4 f = texelFetch(u_flagTex, t, 0);
    return (abs(f.r) + abs(f.g) + abs(f.b) + abs(f.a) > 0.5) ? 1.0 : 0.0;
}

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);

    vec3 a = texelFetch(u_inTex, base, 0).rgb;
    vec3 b = texelFetch(u_baseTex, base, 0).rgb;
    vec3 rgbIn = mix(a, b, u_beta);
    vec3 yccIn = yccOf(rgbIn);

    // u_sigmaTex R holds sigma_hat^2 in raw-DN^2 (10-bit sensor units). Stage-5 input
    // (demosaic FBO) carries sensor units divided down by (whiteLevel-blackLevel),
    // so the variance must be converted to that same scale: multiply BOTH sigma_hat^2
    // and the demosaic residual variance by 1/(whiteLevel-blackLevel)^2 before feeding epsilon.
    // u_epsScale then rescales the raw-DN^2 noise floor to the actual S3+boxAA
    // residual that this pass sees (S5D: measured var ~2-6e-6 image units vs
    // sigma_hat^2~113 DN^2 -> epsilon was ~30-100x the true residual -> aY~0.02 -> output collapsed
    // to the window mean, brightening + flat 5x5 blocks. Scale brings epsilon to the
    // real residual so aY lands in a denoising-friendly range, not a mean-sink.)
    //
    // LIVE-PREVIEW path: when the S2 sigma_hat grid is gated off (Stage-1/2/3 are
    // capture-only), u_use_iso_sigma=1 substitutes the Stage-0 ISO model
    // sigma_hat^2 = a*signal + b with signal sampled at the owning output texel's luma.
    // The luma here is yccIn.x relative to whiteRange (full sensor capacity), so
    // signal is luma*whiteRange and whiteRange = 1/sqrt(u_inverse_range2).
    float sigma2 = 0.0;
    if (u_use_iso_sigma > 0.5) {
        float whiteRangeI = 1.0 / sqrt(max(u_inverse_range2, 1e-6));
        float signalDN = max(yccIn.x * whiteRangeI, 0.0);
        sigma2 = max(u_iso_model_a * signalDN + u_iso_model_b, 0.0);
    } else {
        ivec2 sb = ivec2(floor((vec2(base) + 0.5) * u_sigmaSize / u_outSize));
        sb = clamp(sb, ivec2(0), ivec2(u_sigmaSize) - ivec2(1));
        sigma2 = max(texelFetch(u_sigmaTex, sb, 0).r, 0.0);
    }
    // Luma keeps its texture: epsilon_Y uses a moderate scale (~1.0-1.4) so aY stays
    // high and luminance is only lightly smoothed.
    float epsY = u_lumaEpsScale * (sigma2 + u_sigma_dm2) * u_inverse_range2 * u_epsScale * u_eps_boost;

    // SWGF (Yin 2019 Alg.1) window selection, softened.  A hard argmin among
    // the 8 discrete side windows maps every output pixel to exactly one mean;
    // at a sharp luma edge (clipped-region rim) the winner map becomes an
    // 8-ray starburst - adjacent pixels whose best window differs land on
    // different window means and the seams read as blocky stain patches
    // radiating from the bright region.  Instead fuse the two closest windows
    // with weights proportional to 1/score^2, so the mixture is continuous across seams.  The
    // distance-based weight is unchanged in spirit: a dark pixel still keeps
    // out of bright windows (their 1/score^2 weight is ~0), no brighten/collapse.
    float bestScore = 1.0e30;
    float sndScore = 1.0e30;
    vec2 bestStat = vec2(0.0);
    vec2 sndStat = vec2(0.0);
    float bestVar = 0.0;
    float sndVar = 0.0;
    int Rwin = max(1, int(round(2.0 * u_win_scale)));
    float fullCount = float((2 * Rwin + 1) * (2 * Rwin + 1));
    bool anyEligible = false;
    for (int k = 0; k < 8; k++) {
        ivec2 c = ivec2(vec2(SW_WIN[k]) * u_win_scale);
        vec3 st = statsAt(base, c);
        // Defect gate: a window whose box lost any tap to the mask (count <
        // full) is disqualified.  With no mask every count is full, so every
        // window stays eligible and the selection is bit-identical.
        if (u_flag_available > 0.5 && st.b < fullCount - 0.5) continue;
        anyEligible = true;
        float v = max(st.g - st.r * st.r, 0.0);
        float score = abs(st.r - yccIn.x) / (v + epsY);
        if (score < bestScore) {
            sndScore = bestScore;
            sndVar = bestVar;
            sndStat = bestStat;
            bestScore = score;
            bestVar = v;
            bestStat = st.xy;
        } else if (score < sndScore) {
            sndScore = score;
            sndVar = v;
            sndStat = st.xy;
        }
    }
    // Every window is disqualified: fall back to the renormalized stats of all
    // windows so a defect-dense pixel degrades gracefully instead of switching
    // the filter off (the masks are sparse, so this is the rare corner).
    if (!anyEligible) {
        for (int k = 0; k < 8; k++) {
            ivec2 c = ivec2(vec2(SW_WIN[k]) * u_win_scale);
            vec3 st = statsAt(base, c);
            float v = max(st.g - st.r * st.r, 0.0);
            float score = abs(st.r - yccIn.x) / (v + epsY);
            if (score < bestScore) {
                sndScore = bestScore;
                sndVar = bestVar;
                sndStat = bestStat;
                bestScore = score;
                bestVar = v;
                bestStat = st.xy;
            } else if (score < sndScore) {
                sndScore = score;
                sndVar = v;
                sndStat = st.xy;
            }
        }
    }
    float wBest = 1.0 / (bestScore * bestScore + 1e-12);
    float wSnd = 1.0 / (sndScore * sndScore + 1e-12);
    float wTot = wBest + wSnd;
    float pBest = wBest / wTot;
    float pSnd = wSnd / wTot;
    float meanY = pBest * bestStat.r + pSnd * sndStat.r;
    float varY = pBest * bestVar + pSnd * sndVar;

    float aY = varY / (varY + epsY);
    float outY = aY * yccIn.x + (1.0 - aY) * meanY;

    // The coherence below also weights the grain gate's return, and rides the
    // output alpha to get there: a return added where the estimate was cut by a
    // boundary is an echo of that boundary, and this test is the stage's own
    // answer to where that is. Nothing else reads the alpha. 1.0 is the gate's
    // shipped weight, so a path that skips the test leaves the gate untouched.
    float gateW = 1.0;

    // Support shrinkage. The centre window's variance carries whatever
    // structure the box holds on top of the noise, so its excess over the
    // flat-patch level measures how much of the window is a boundary rather
    // than a surface. Where it is cut by one, the wide estimate is biased
    // across the boundary - the halo beside a high-contrast edge, the fringe
    // along a clipped region - so the pass falls back to a 3x3 mean: it gives
    // up the wide window's smoothing there instead of moving the boundary.
    // The narrower support also stops the fallback spreading a bright edge
    // about 2 px into a dark surround.
    if (u_shrink > 0.5 && Rwin > 2) {
        vec3 stc = statsAt(base, ivec2(0));
        float varC = max(stc.g - stc.r * stc.r, 0.0);
        float ref2 = max(u_shrink_ref * epsY, 1e-30);
        float coh = clamp(1.0 - (varC - ref2) / (u_shrink_tau * ref2), 0.0, 1.0);
        coh = min(coh, u_shrink_cap);
        // The cap is 0.5, so the double normalises the gate's weight to 1 where
        // the window is whole and to 0 where the boundary owns it.
        gateW = clamp(2.0 * coh, 0.0, 1.0);
        // 3x3 fallback from nine taps: the centre (in hand) and the eight
        // neighbours. The narrow support is the point; the tap count is the
        // cost of the same support on the preview and the capture path.
        int r2 = 1;
        int h2 = 1;
        float smallY = yccIn.x;
        smallY += yccOf(mix(texelFetch(u_inTex, base + ivec2( r2,   0), 0).rgb,
                            texelFetch(u_baseTex, base + ivec2( r2,   0), 0).rgb, u_beta)).x;
        smallY += yccOf(mix(texelFetch(u_inTex, base + ivec2(-r2,   0), 0).rgb,
                            texelFetch(u_baseTex, base + ivec2(-r2,   0), 0).rgb, u_beta)).x;
        smallY += yccOf(mix(texelFetch(u_inTex, base + ivec2(  0,  r2), 0).rgb,
                            texelFetch(u_baseTex, base + ivec2(  0,  r2), 0).rgb, u_beta)).x;
        smallY += yccOf(mix(texelFetch(u_inTex, base + ivec2(  0, -r2), 0).rgb,
                            texelFetch(u_baseTex, base + ivec2(  0, -r2), 0).rgb, u_beta)).x;
        smallY += yccOf(mix(texelFetch(u_inTex, base + ivec2( h2,  h2), 0).rgb,
                            texelFetch(u_baseTex, base + ivec2( h2,  h2), 0).rgb, u_beta)).x;
        smallY += yccOf(mix(texelFetch(u_inTex, base + ivec2(-h2,  h2), 0).rgb,
                            texelFetch(u_baseTex, base + ivec2(-h2,  h2), 0).rgb, u_beta)).x;
        smallY += yccOf(mix(texelFetch(u_inTex, base + ivec2( h2, -h2), 0).rgb,
                            texelFetch(u_baseTex, base + ivec2( h2, -h2), 0).rgb, u_beta)).x;
        smallY += yccOf(mix(texelFetch(u_inTex, base + ivec2(-h2, -h2), 0).rgb,
                            texelFetch(u_baseTex, base + ivec2(-h2, -h2), 0).rgb, u_beta)).x;
        outY = mix(smallY / 9.0, outY, coh);
    }

    // Chroma is bit-exact passthrough: S5-L filters luma only. The chroma
    // branch lives in S5-C (S5CShaderProgram) and reads this pass's luma as
    // its guide.
    outColor = vec4(mix(rgbIn, rgbOf(vec3(outY, yccIn.y, yccIn.z)), u_strength), gateW);
}
"""

        private fun createProgram(vertexSource: String, fragmentSource: String): Int {
            // Inject the window-centre constant array into the main shader.
            val frag = if (fragmentSource == MAIN_FRAGMENT) {
                val sb = StringBuilder()
                sb.append("#version 300 es\n")
                sb.append("const ivec2 SW_WIN[8] = ivec2[8](")
                for ((i, c) in WINDOW_CENTERS.withIndex()) {
                    if (i > 0) sb.append(", ")
                    sb.append("ivec2(${c[0]}, ${c[1]})")
                }
                sb.append(");\n")
                sb.append(fragmentSource)
                sb.toString()
            } else {
                fragmentSource
            }
            val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource)
            val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, frag)
            if (vertexShader == 0 || fragmentShader == 0) return 0

            val program = GLES20.glCreateProgram()
            GLES20.glAttachShader(program, vertexShader)
            GLES20.glAttachShader(program, fragmentShader)
            GLES20.glLinkProgram(program)

            val linkStatus = IntArray(1)
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0)
            if (linkStatus[0] != GLES20.GL_TRUE) {
                val info = "Program link failed: ${GLES20.glGetProgramInfoLog(program)}"
                Log.e(TAG, info)
                com.agx.camera.CrashLogger.log(TAG, info)
                GLES20.glDeleteProgram(program)
                return 0
            }
            GLES20.glDeleteShader(vertexShader)
            GLES20.glDeleteShader(fragmentShader)
            return program
        }

        private fun loadShader(type: Int, source: String): Int {
            val shader = GLES20.glCreateShader(type)
            GLES20.glShaderSource(shader, source)
            GLES20.glCompileShader(shader)
            val compiled = IntArray(1)
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
            if (compiled[0] != GLES20.GL_TRUE) {
                val info = "Shader compile failed: ${GLES20.glGetShaderInfoLog(shader)}"
                Log.e(TAG, info)
                com.agx.camera.CrashLogger.log(TAG, info)
                GLES20.glDeleteShader(shader)
                return 0
            }
            return shader
        }
    }
}