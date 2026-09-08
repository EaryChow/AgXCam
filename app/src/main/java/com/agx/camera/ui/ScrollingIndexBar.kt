package com.agx.camera.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Typeface
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import com.agx.camera.CrashLogger
import kotlin.math.ceil

/**
 * Pure scroll-to-index mapping used by [ScrollingIndexBar].
 *
 * A drag is converted into whole detents of `spacingPx`; the selection index steps by one
 * per detent (haptic tick in the view) and the remainder is carried back so the strip
 * glides continuously with the finger and is always re-centered on the selected index at
 * each detent boundary.
 */
class ScrollIndexModel(val maxIndex: Int, val spacingPx: Float) {
    var index: Int = 0
        private set
    var shiftPx: Float = 0f
        private set

    private var gestureAccumPx = 0f
    private var gestureBaseIndex = 0

    init {
        require(maxIndex >= 0) { "maxIndex must be >= 0" }
        index = (maxIndex / 2).coerceIn(0, maxIndex)
    }

    fun setIndex(value: Int) {
        index = value.coerceIn(0, maxIndex)
        shiftPx = 0f
    }

    fun beginGesture() {
        gestureBaseIndex = index
        gestureAccumPx = 0f
        shiftPx = 0f
    }

    /** Consume a vertical drag delta (screen px, +down) and return the new selected index. */
    fun consumeDrag(dyPx: Float): Int {
        if (spacingPx <= 0f || maxIndex <= 0) return index
        gestureAccumPx += dyPx
        val shift = -gestureAccumPx
        val steps = Math.round(shift / spacingPx)
        index = (gestureBaseIndex + steps).coerceIn(0, maxIndex)
        shiftPx = shift - steps * spacingPx
        return index
    }
}

/**
 * Vertical scrolling index picker with a fixed center dot.
 *
 * The strip stays the same size as the slider line it replaces; the dot is fixed in the
 * middle and the number strip scrolls underneath. Numbers outside the strip's end points
 * are not drawn. Each detent (index change) triggers a haptic tick.
 *
 * Labels are drawn with [perTextRotation] applied around each label's own center so the
 * text can be re-oriented independently of the (unchanged) strip graphics in the future.
 */
class ScrollingIndexBar @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var maxIndex: Int = 0
        set(value) {
            val v = value.coerceAtLeast(0)
            if (v != field) {
                val prevIndex = model.index
                field = v
                model = ScrollIndexModel(v, spacingPx)
                model.setIndex(prevIndex)
                if (resetIndex < 0 || resetIndex > v) resetIndex = v / 2
            }
            invalidate()
        }

    val index: Int
        get() = model.index

    var spacingPx: Float = DEFAULT_SPACING_DP * resources.displayMetrics.density
        set(value) {
            val v = value.coerceAtLeast(1f)
            if (v != field) {
                val prevIndex = model.index
                field = v
                model = ScrollIndexModel(maxIndex, v)
                model.setIndex(prevIndex)
            }
            invalidate()
        }

    var labelFormatter: (Int) -> String = { it.toString() }

    /** Called whenever the selected index changes during use (scroll or reset). */
    var onIndexChange: ((Int) -> Unit)? = null

    /** Whether each detent (index change) triggers a haptic tick. */
    var hapticEnabled: Boolean = true

    /** Double-tap resets to this index; -1 disables the double-tap reset. */
    var resetIndex: Int = -1

    /** Rotate every label around its own center by this angle (degrees). */
    var perTextRotation: Float = 0f

    private var model: ScrollIndexModel = ScrollIndexModel(maxIndex, spacingPx)

    private val tickPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        setShadowLayer(3f * resources.displayMetrics.density, 0f, 0f, 0x66000000.toInt())
    }

    private val touchSlop = ViewConfiguration.get(context).scaledTouchSlop
    private var lastTapTime = 0L
    private var lastTapX = 0f
    private var lastTapY = 0f
    private var lastDragY = 0f
    private var lastNotifiedIndex = 0
    private var tickCalls = 0
    private var vibDiagLogged = false

    private val tickVibrator: Vibrator? by lazy { tickVibratorFor(context) }

    private var lastTickTime = 0L

    /** Instrumented vibrate path: every detent attempt is logged to CrashLogger (the
     *  capture that actually shows up in the saved debug file) so the code path, the
     *  throttling, and the vibrator state are all observable. */
    private fun vibrateTick() {
        tickCalls++
        if (VIBE_DELAY_MS > 0L) {
            postDelayed(::vibrateTickNow, VIBE_DELAY_MS)
        } else {
            vibrateTickNow()
        }
    }

    private fun vibrateTickNow() {
        tickCalls++
        val v = tickVibrator
        if (v == null || !v.hasVibrator()) {
            CrashLogger.log(TAG, "TICK calls=$tickCalls no-vibrator=${tickVibrator == null}")
            return
        }
        // A new vibrate() cancels the previous one. During a fast scroll detents arrive
        // quicker than the motor can ramp, so throttle to let each pulse complete.
        val now = android.os.SystemClock.uptimeMillis()
        if (now - lastTickTime < MIN_TICK_INTERVAL_MS) {
            CrashLogger.log(TAG, "TICK calls=$tickCalls throttle-skip idx=${model.index} dt=${now - lastTickTime}ms")
            return
        }
        lastTickTime = now
        try {
            v.vibrate(VibrationEffect.createOneShot(TICK_MS, TICK_AMPLITUDE))
            CrashLogger.log(TAG, "TICK calls=$tickCalls vibrate idx=${model.index} ms=$TICK_MS amp=$TICK_AMPLITUDE")
        } catch (e: Exception) {
            CrashLogger.log(TAG, "TICK calls=$tickCalls FAIL ${e.javaClass.simpleName}: ${e.message}")
            @Suppress("DEPRECATION")
            v.vibrate(TICK_MS)
        }
        if (!vibDiagLogged) {
            vibDiagLogged = true
            CrashLogger.log(
                TAG,
                "TICK DIAG vibrator=${v.javaClass.simpleName} hasAmpCtrl=${v.hasAmplitudeControl()} sdk=${Build.VERSION.SDK_INT}"
            )
        }
    }

    init {
        isHapticFeedbackEnabled = true
    }

    /** Re-center the strip on the current selection (used when the popup is opened). */
    fun recenter() {
        setIndex(model.index)
    }

    /** Set the selected index and re-center the strip on it. */
    fun setIndex(value: Int) {
        model.setIndex(value)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val spacing = model.spacingPx
        if (spacing <= 0f) return

        val density = resources.displayMetrics.density
        val centerY = height / 2f
        val lineX = width / 2f
        val visibleCount = (ceil((height / 2f) / spacing).toInt()) + 1

        for (offset in -visibleCount..visibleCount) {
            val i = model.index + offset
            if (i < 0 || i > maxIndex) continue
            val y = centerY + offset * spacing - model.shiftPx
            val selected = offset == 0

            tickPaint.color = if (selected) 0xFFFFFFFF.toInt() else 0x30FFFFFF.toInt()
            tickPaint.strokeWidth = if (selected) 2.5f * density else 2f * density
            canvas.drawRect(lineX + 2f * density, y - 1f * density, lineX + 9f * density, y + 1f * density, tickPaint)

            val label = labelFormatter(i)
            if (label.isEmpty()) continue

            textPaint.textSize = (if (selected) CENTER_TEXT_SP else TEXT_SP) * density
            textPaint.color = if (selected) 0xFFFFFFFF.toInt() else 0x88FFFFFF.toInt()
            textPaint.typeface = if (selected) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
            val fm = textPaint.fontMetrics
            val halfHeight = (fm.descent - fm.ascent) / 2f
            if (y + halfHeight < 0f || y - halfHeight > height) continue

            val labelX = lineX + 12f * density
            val labelCenterX = labelX + textPaint.measureText(label) / 2f
            val baseline = y - (fm.ascent + fm.descent) / 2f
            val labelCenterY = baseline - (fm.ascent + fm.descent) / 2f
            canvas.save()
            canvas.rotate(perTextRotation, labelCenterX, labelCenterY)
            canvas.drawText(label, labelX, baseline, textPaint)
            canvas.restore()
        }

        canvas.drawCircle(lineX, centerY, 6f * density, dotPaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!isEnabled) return false
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                if (resetIndex >= 0 &&
                    event.eventTime - lastTapTime < ViewConfiguration.getDoubleTapTimeout() &&
                    kotlin.math.abs(event.x - lastTapX) < touchSlop &&
                    kotlin.math.abs(event.y - lastTapY) < touchSlop
                ) {
                    model.setIndex(resetIndex)
                    if (hapticEnabled) vibrateTick()
                    onIndexChange?.invoke(resetIndex)
                    invalidate()
                    return true
                }
                lastTapTime = event.eventTime
                lastTapX = event.x
                lastTapY = event.y
                lastDragY = event.y
                model.beginGesture()
                lastNotifiedIndex = model.index
                parent?.requestDisallowInterceptTouchEvent(true)
                CrashLogger.log(TAG, "DOWN touch index=${model.index} range=0..$maxIndex")
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val dy = event.y - lastDragY
                lastDragY = event.y
                val newIndex = model.consumeDrag(dy)
                if (newIndex != lastNotifiedIndex) {
                    lastNotifiedIndex = newIndex
                    if (hapticEnabled) vibrateTick()
                    onIndexChange?.invoke(newIndex)
                }
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                parent?.requestDisallowInterceptTouchEvent(false)
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    companion object {
        const val DEFAULT_SPACING_DP = 24f
        const val TEXT_SP = 9f
        const val CENTER_TEXT_SP = 11f
        const val TICK_MS = 18L
        const val TICK_AMPLITUDE = 100
        const val MIN_TICK_INTERVAL_MS = 50L
        const val VIBE_DELAY_MS = 130L
        private const val TAG = "ScrollingIndexBar"

        fun tickVibratorFor(context: Context): Vibrator? {
            return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val manager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
                manager?.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
            }
        }
    }
}