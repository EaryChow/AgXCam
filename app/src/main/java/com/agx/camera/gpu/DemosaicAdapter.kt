package com.agx.camera.gpu

import android.opengl.GLES30
import android.opengl.GLES31
import com.agx.camera.camera.BayerPattern

interface DemosaicAdapter {

    val name: String

    val strategy: String

    fun createComputeShaderSource(
        bayerPattern: BayerPattern,
        bitDepth: Int
    ): String

    fun tileSize(): Int = 2048

    fun overlapPixels(): Int = 2

    fun supported(): Boolean = true

    /**
     * Sets tile-specific uniforms for the compute shader.
     * Must be called before glDispatchCompute for each tile.
     *
     * IMPORTANT: u_tileSize MUST be set to (tile.width, tile.height) — the actual
     * FBO dimensions including overlap — NOT the base tile size. The shader uses
     * u_tileSize as the write bounds; setting it to the base size leaves the
     * overlap border uninitialized.
     */
    fun setTileUniforms(
        programId: Int,
        tile: TiledComputeScheduler.Tile,
        sensorWidth: Int,
        sensorHeight: Int
    ) {
        val uTileOrigin = GLES30.glGetUniformLocation(programId, "u_tileOrigin")
        val uTileSize = GLES30.glGetUniformLocation(programId, "u_tileSize")
        val uSensorSize = GLES30.glGetUniformLocation(programId, "u_sensorSize")
        GLES30.glUniform2i(uTileOrigin, tile.originX, tile.originY)
        GLES30.glUniform2i(uTileSize, tile.width, tile.height)
        GLES30.glUniform2i(uSensorSize, sensorWidth, sensorHeight)
    }

    /**
     * Dispatches the compute shader for a single tile.
     * Called by TiledComputeScheduler.executeTiledPipeline via the dispatchTile callback.
     * Binds the Bayer texture, sets uniforms, and issues glDispatchCompute.
     */
    fun dispatchCompute(
        programId: Int,
        tile: TiledComputeScheduler.Tile,
        bayerTextureId: Int,
        outputTextureId: Int,
        sensorWidth: Int,
        sensorHeight: Int,
        bitDepth: Int
    ) {
        GLES30.glUseProgram(programId)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, bayerTextureId)

        GLES31.glBindImageTexture(1, outputTextureId, 0, false, 0,
            GLES31.GL_WRITE_ONLY, GLES30.GL_RGBA16F)

        setTileUniforms(programId, tile, sensorWidth, sensorHeight)

        val groupsX = (tile.width + 15) / 16
        val groupsY = (tile.height + 15) / 16
        GLES31.glDispatchCompute(groupsX, groupsY, 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    }
}

class StandardBayerAdapter : DemosaicAdapter {

    override val name: String = "standard_bayer"
    override val strategy: String = "edge_directed"

    override fun supported(): Boolean = true

    override fun createComputeShaderSource(
        bayerPattern: BayerPattern,
        bitDepth: Int
    ): String {
        val colorMap = bayerPattern.colorMap()
        return computeTemplate(colorMap, bitDepth)
    }

    private fun computeTemplate(colorMap: IntArray, bitDepth: Int): String {
        val mask = when {
            bitDepth <= 10 -> "0x3FFu"
            bitDepth <= 12 -> "0xFFFu"
            bitDepth <= 14 -> "0x3FFFu"
            else -> "0xFFFFu"
        }
        return """
#version 310 es
layout(local_size_x = 16, local_size_y = 16) in;
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

layout(binding = 0) uniform highp usampler2D u_bayerTex;
layout(rgba16f, binding = 1) writeonly uniform highp image2D u_outTex;

uniform ivec4 u_black_level_pattern;
uniform ivec4 u_bayer_color_map;
uniform sampler2D u_lens_shading_map;
uniform int u_bit_depth;
uniform ivec2 u_sensorSize;
uniform ivec2 u_tileOrigin;
uniform ivec2 u_tileSize;

uint unpackRaw(uint rawPacked) {
    uint mask = ${mask}u;
    return rawPacked & mask;
}

float sampleBayer(ivec2 coord) {
    ivec2 clamped = clamp(coord, ivec2(0), u_sensorSize - ivec2(1));
    uint raw = texelFetch(u_bayerTex, clamped, 0).r;
    float val = float(unpackRaw(raw));
    int phase = (clamped.x % 2) + (clamped.y % 2) * 2;
    val -= float(u_black_level_pattern[phase]);
    return max(val, 0.0);
}

float lensGain(ivec2 coord) {
    ivec2 clamped = clamp(coord, ivec2(0), u_sensorSize - ivec2(1));
    vec2 uv = vec2(clamped) / vec2(u_sensorSize);
    vec4 gains = texture(u_lens_shading_map, uv);
    int phase = abs(clamped.x % 2) + abs(clamped.y % 2) * 2;
    return gains[phase];
}

float sb(ivec2 coord) {
    return sampleBayer(coord) * lensGain(coord);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= u_tileSize.x || pixel.y >= u_tileSize.y) return;

    ivec2 pos = pixel + u_tileOrigin;
    int phase = abs(pos.x % 2) + abs(pos.y % 2) * 2;
    int color = u_bayer_color_map[phase];

    float gH  = lensGain(pos + ivec2(-2, 0));
    float gH2 = lensGain(pos + ivec2( 2, 0));
    float gV  = lensGain(pos + ivec2( 0,-2));
    float gV2 = lensGain(pos + ivec2( 0, 2));

    float dH = abs(sampleBayer(pos + ivec2(-2, 0)) * gH
                 - sampleBayer(pos + ivec2( 2, 0)) * gH2);
    float dV = abs(sampleBayer(pos + ivec2( 0,-2)) * gV
                 - sampleBayer(pos + ivec2( 0, 2)) * gV2);

    float wH = max(4.0 * dH - dV, 0.0);
    float wV = max(4.0 * dV - dH, 0.0);

    float r = 0.0, g = 0.0, b = 0.0;

    if (color == 0) {
        r = sb(pos);
        float gAvg = (sb(pos + ivec2( 0,-1)) + sb(pos + ivec2( 0, 1))
                    + sb(pos + ivec2(-1, 0)) + sb(pos + ivec2( 1, 0))) * 0.25;
        float rLap = 4.0 * r
            - sampleBayer(pos + ivec2(-2, 0)) * gH
            - sampleBayer(pos + ivec2( 2, 0)) * gH2
            - sampleBayer(pos + ivec2( 0,-2)) * gV
            - sampleBayer(pos + ivec2( 0, 2)) * gV2;
        g = gAvg - rLap * 0.125;

        float bNW = sb(pos + ivec2(-1,-1));
        float bNE = sb(pos + ivec2( 1,-1));
        float bSW = sb(pos + ivec2(-1, 1));
        float bSE = sb(pos + ivec2( 1, 1));
        b = (bNW + bNE + bSW + bSE) * 0.25 + rLap * 0.125;

    } else if (color == 2) {
        b = sb(pos);
        float gAvg = (sb(pos + ivec2( 0,-1)) + sb(pos + ivec2( 0, 1))
                    + sb(pos + ivec2(-1, 0)) + sb(pos + ivec2( 1, 0))) * 0.25;
        float bLap = 4.0 * b
            - sampleBayer(pos + ivec2(-2, 0)) * gH
            - sampleBayer(pos + ivec2( 2, 0)) * gH2
            - sampleBayer(pos + ivec2( 0,-2)) * gV
            - sampleBayer(pos + ivec2( 0, 2)) * gV2;
        g = gAvg - bLap * 0.125;

        float rNW = sb(pos + ivec2(-1,-1));
        float rNE = sb(pos + ivec2( 1,-1));
        float rSW = sb(pos + ivec2(-1, 1));
        float rSE = sb(pos + ivec2( 1, 1));
        r = (rNW + rNE + rSW + rSE) * 0.25 + bLap * 0.125;

    } else {
        g = sb(pos);
        int colorNS = u_bayer_color_map[abs(pos.x % 2) + abs((pos.y - 1) % 2) * 2];

        if (colorNS == 0) {
            float rAvg = (sb(pos + ivec2( 0,-1)) + sb(pos + ivec2( 0, 1))) * 0.5;
            float gCorr = 2.0 * g
                - sampleBayer(pos + ivec2( 0,-2)) * gV
                - sampleBayer(pos + ivec2( 0, 2)) * gV2;
            r = rAvg + gCorr * 0.25;

            float bAvg = (sb(pos + ivec2(-1, 0)) + sb(pos + ivec2( 1, 0))) * 0.5;
            float gCorr2 = 2.0 * g
                - sampleBayer(pos + ivec2(-2, 0)) * gH
                - sampleBayer(pos + ivec2( 2, 0)) * gH2;
            b = bAvg + gCorr2 * 0.25;
        } else {
            float bAvg = (sb(pos + ivec2( 0,-1)) + sb(pos + ivec2( 0, 1))) * 0.5;
            float gCorr = 2.0 * g
                - sampleBayer(pos + ivec2( 0,-2)) * gV
                - sampleBayer(pos + ivec2( 0, 2)) * gV2;
            b = bAvg + gCorr * 0.25;

            float rAvg = (sb(pos + ivec2(-1, 0)) + sb(pos + ivec2( 1, 0))) * 0.5;
            float gCorr2 = 2.0 * g
                - sampleBayer(pos + ivec2(-2, 0)) * gH
                - sampleBayer(pos + ivec2( 2, 0)) * gH2;
            r = rAvg + gCorr2 * 0.25;
        }
    }

    imageStore(u_outTex, pixel, vec4(r, g, b, 1.0));
}
"""
    }
}

class PreviewBilinearDemosaicAdapter : DemosaicAdapter {

    override val name: String = "preview_bilinear"
    override val strategy: String = "sparse_bilinear"

    override fun supported(): Boolean = true

    override fun tileSize(): Int = 0

    override fun overlapPixels(): Int = 0

    override fun createComputeShaderSource(
        bayerPattern: BayerPattern,
        bitDepth: Int
    ): String = ""
}
