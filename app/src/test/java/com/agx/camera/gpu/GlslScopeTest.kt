package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Scope resolution for the GLSL this project ships.
 *
 * The reason this exists: a P1 shader variant was written, passed the whole JVM
 * suite, and failed to compile on a real driver with `'p' undeclared
 * identifier`. Every test in the repo read the shader as text and asserted on
 * substrings, and not one of them could tell a declared name from an undeclared
 * one, because a substring assertion does not know what scope it is in. So the
 * bug that shipped was invisible to the entire gate.
 *
 * This is not a GLSL compiler and does not pretend to be one. It resolves one
 * thing - whether an identifier used in a function body was declared in that
 * function, at global scope, or is a builtin - which is exactly the failure that
 * got through. It will not catch type errors, and it says so in the output
 * rather than reporting a clean bill of health it has not earned.
 *
 * The self-test at the bottom is what keeps this honest. A resolver that finds
 * nothing is indistinguishable from a resolver that checks nothing, so the
 * historical defect is re-injected and the resolver is required to catch it.
 */
class GlslScopeTest {

    private fun source(name: String): String {
        val f = File("src/main/java/com/agx/camera/gpu/$name.kt")
        assertTrue("cannot read $name.kt from ${f.absolutePath}", f.exists())
        return f.readText()
    }

    /** Pulls a `private const val NAME = """ ... """` block out of a Kotlin file. */
    private fun rawString(kotlin: String, name: String): String {
        val start = kotlin.indexOf("""$name = """")
        assertTrue("$name is not declared in this file", start >= 0)
        val open = kotlin.indexOf("\"\"\"", start)
        val close = kotlin.indexOf("\"\"\"", open + 3)
        assertTrue("$name has no closed raw string", open >= 0 && close > open)
        return kotlin.substring(open + 3, close)
    }

    private val typeWords = setOf(
        "void", "bool", "int", "uint", "float", "double",
        "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4",
        "bvec2", "bvec3", "bvec4", "mat2", "mat3", "mat4",
        "sampler2D", "sampler2DArray", "sampler3D"
    )

    private val builtins = setOf(
        // Constructors and casts.
        "float", "int", "uint", "bool", "vec2", "vec3", "vec4",
        "ivec2", "ivec3", "ivec4", "bvec2", "bvec3", "bvec4",
        "mat2", "mat3", "mat4",
        // Angle and trigonometry.
        "radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "pow", "exp", "log", "exp2", "log2",
        "sqrt", "inversesqrt",
        // Common.
        "abs", "sign", "floor", "trunc", "round", "roundEven", "ceil", "fract",
        "mod", "modf", "min", "max", "clamp", "mix", "step", "smoothstep",
        "isnan", "isinf", "floatBitsToInt", "intBitsToFloat",
        // Vector.
        "length", "distance", "dot", "cross", "normalize", "faceforward",
        "reflect", "refract", "component",
        // Texture.
        "texture", "textureLod", "texelFetch", "textureSize", "textureProj",
        // Integer.
        "bitfieldExtract", "bitfieldInsert", "bitfieldReverse", "bitCount",
        "findLSB", "findMSB", "uaddCarry", "ushr", "ushl",
        // Derivative, matrix, vector-length.
        "dFdx", "dFdy", "fwidth", "transpose", "inverse", "determinant",
        // Geometry.
        "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual",
        "equal", "notEqual", "any", "all", "not",
        // Builtin variables.
        "gl_FragCoord", "gl_Position", "gl_PointSize", "gl_VertexID",
        "gl_InstanceID", "gl_FrontFacing", "gl_PointCoord",
        // Control flow keywords that can appear where an identifier would.
        "return", "if", "else", "for", "while", "do", "break", "continue",
        "discard", "const", "uniform", "in", "out", "inout", "layout",
        "precision", "highp", "mediump", "lowp", "struct", "switch", "case",
        "default", "true", "false"
    )

    /**
     * Finds every identifier a function body uses without having declared it.
     *
     * Returns the offending names, empty when the body resolves. Declarations
     * are consumed first so a name is only reported when it is read, not when
     * it is introduced.
     */
    private fun undeclaredIn(
        body: String,
        globals: Set<String>,
        params: Set<String>,
        functionNames: Set<String>
    ): Set<String> {
        // Strip comments so a name mentioned only in prose is never reported,
        // then strip numeric literals. The second one is not cosmetic: 1.0e-6
        // and 1.0e30 both contain a bare 'e', which reads as an undeclared
        // identifier and would bury the real findings under noise.
        val clean = body
            .replace(Regex("//[^\\n]*"), " ")
            .replace(Regex("/\\*.*?\\*/", RegexOption.DOT_MATCHES_ALL), " ")
            .replace(Regex("\\b\\d+\\.?\\d*[eE][+-]?\\d+f?\\b"), " ")
            .replace(Regex("\\b\\d+\\.?\\d*[fFuU]?\\b"), " ")

        val declared = HashSet(globals)
        declared.addAll(params)
        val declPattern = Regex("\\b(${typeWords.joinToString("|")})\\s+([A-Za-z_]\\w*)")
        for (m in declPattern.findAll(clean)) declared.add(m.groupValues[2])

        // Blank out the declaration sites so what remains is genuinely a use.
        val uses = declPattern.replace(clean) { " ".repeat(it.value.length) }

        val bad = LinkedHashSet<String>()
        val ident = Regex("[A-Za-z_]\\w*")
        for (m in ident.findAll(uses)) {
            val name = m.value
            if (name in declared || name in builtins || name in functionNames) continue
            // A field or swizzle reads the member, not a bare name.
            val before = uses.substring(0, m.range.first).trimEnd()
            if (before.endsWith(".")) continue
            // A call is a function reference, and user functions are known.
            val after = uses.substring(m.range.last + 1).trimStart()
            if (after.startsWith("(")) continue
            bad.add(name)
        }
        return bad
    }

    /** Splits a shader into (name, params, body), brace-counted rather than guessed. */
    private fun functions(src: String): List<Triple<String, Set<String>, String>> {
        val out = ArrayList<Triple<String, Set<String>, String>>()
        val header = Regex("\\b(?:void|float|int|vec[234]|ivec[234]|bool|mat[234])\\s+([A-Za-z_]\\w*)\\s*\\(")
        for (m in header.findAll(src)) {
            val paramOpen = src.indexOf('(', m.range.first)
            if (paramOpen < 0) continue
            // Balance the parameter list. It is legal for a default-looking
            // comma-free array parameter to contain no commas at all, so this
            // cannot be split on "," alone.
            var pDepth = 0
            var pi = paramOpen
            while (pi < src.length) {
                when (src[pi]) {
                    '(' -> pDepth++
                    ')' -> {
                        pDepth--
                        if (pDepth == 0) break
                    }
                }
                pi++
            }
            if (pDepth != 0) continue
            val paramText = src.substring(paramOpen + 1, pi)
            val open = src.indexOf('{', pi)
            if (open < 0) continue
            var depth = 0
            var i = open
            while (i < src.length) {
                when (src[i]) {
                    '{' -> depth++
                    '}' -> {
                        depth--
                        if (depth == 0) break
                    }
                }
                i++
            }
            if (depth != 0) continue
            val params = LinkedHashSet<String>()
            for (p in Regex("\\b(?:void|bool|int|uint|float|vec[234]|ivec[234]|bvec[234]|mat[234]|sampler2D)\\s+([A-Za-z_]\\w*)")
                .findAll(paramText)) {
                params.add(p.groupValues[1])
            }
            out.add(Triple(m.groupValues[1], params, src.substring(open + 1, i)))
        }
        return out
    }

    private fun globalsOf(src: String): MutableSet<String> {
        val names = LinkedHashSet<String>()
        val pattern = Regex("\\b(?:uniform|in|out|const)\\s+(?:(?:highp|mediump|lowp)\\s+)?(?:[A-Za-z_]\\w*)\\s+([A-Za-z_]\\w*)")
        for (m in pattern.findAll(src)) names.add(m.groupValues[1])
        // Array uniforms declare their name before the bracket.
        val array = Regex("\\b(?:uniform|const)\\s+(?:(?:highp|mediump|lowp)\\s+)?(?:[A-Za-z_]\\w*)\\s+([A-Za-z_]\\w*)\\s*\\[")
        for (m in array.findAll(src)) names.add(m.groupValues[1])
        return names
    }

    /**
     * Constants the host splices into the source before compiling, read out of
     * the host rather than listed here. The main fragment shader gets its window
     * centres this way, so they are legitimately in scope on the GPU and
     * legitimately absent from the raw string - hardcoding the names would let
     * the list drift from what the host actually emits.
     */
    private fun injectedGlobals(kotlin: String): Set<String> {
        val names = LinkedHashSet<String>()
        for (m in Regex("""append\(\s*"const\s+\w+\s+([A-Za-z_]\w*)""").findAll(kotlin)) {
            names.add(m.groupValues[1])
        }
        return names
    }

    private fun resolve(src: String, label: String, injected: Set<String> = emptySet()): Set<String> {
        val globals = globalsOf(src)
        globals.addAll(injected)
        val fns = functions(src)
        val fnNames = fns.map { it.first }.toSet()
        val bad = LinkedHashSet<String>()
        for ((name, params, body) in fns) {
            for (u in undeclaredIn(body, globals, params, fnNames)) bad.add("$label::$name -> $u")
        }
        return bad
    }

    // ---- the actual gate -------------------------------------------------

    @Test
    fun everyChangedShaderResolves() {
        // The three programs touched in this phase. Each is compiled on a real
        // driver at startup and a failure there is a black screen, so it is
        // worth catching here where it costs seconds.
        val programs = listOf(
            "StructureMapShaderProgram" to listOf("VERTEX_SHADER", "FRAGMENT_SHADER"),
            "OutNrShaderProgram" to listOf("VERTEX_SHADER", "MAIN_FRAGMENT")
        )
        val failures = ArrayList<String>()
        for ((program, blocks) in programs) {
            val kotlin = source(program)
            val injected = injectedGlobals(kotlin)
            for (block in blocks) {
                val src = runCatching { rawString(kotlin, block) }.getOrNull()
                if (src == null) {
                    failures.add("$program.$block: not present, cannot resolve")
                    continue
                }
                failures.addAll(resolve(src, "$program.$block", injected))
            }
        }
        assertEquals(
            "every identifier must resolve in the scope it is used in:\n" +
                failures.joinToString("\n") { "  $it" },
            0, failures.size
        )
    }

    // ---- the self-test ---------------------------------------------------

    @Test
    fun theResolverActuallyCatchesTheBugThatGotThrough() {
        // A resolver that finds nothing is indistinguishable from one that
        // checks nothing. Re-inject the exact defect that shipped and require it
        // to be reported, so "no undeclared identifiers" above means something.
        val src = rawString(source("StructureMapShaderProgram"), "FRAGMENT_SHADER")
        assertTrue(
            "the fixed shader must not already contain the defect",
            src.contains("float ratio = localRatio(px);")
        )
        val broken = src.replace(
            "float ratio = localRatio(px);",
            "float ratio = localRatio(p);"
        )
        assertTrue("the injected defect did not change the source", broken != src)

        val caught = resolve(broken, "injected")
        assertTrue(
            "the resolver missed the historical defect: $caught",
            caught.any { it.endsWith("-> p") }
        )
    }

    // ---- host bindings ---------------------------------------------------

    @Test
    fun everyUniformTheMapBindsIsLocated() {
        // Uniform values are runtime, so a missing binding does not fail to
        // compile and silently reads as a dead control. Each uniform the map
        // uses is asserted to be located on the host.
        val map = source("StructureMapShaderProgram")
        for (uniform in listOf(
            "u_luma_eps_scale", "u_sigma_dm2", "u_inverse_range2", "u_calib_scale", "u_eps_boost"
        )) {
            assertTrue(
                "$uniform must be located on the host",
                map.contains("glGetUniformLocation(programId, \"$uniform\")")
            )
        }
        // The map is a census-only measurement: the host must not have kept a
        // binding for an S5-side consumer, because there is no S5 side.
        for (gone in listOf("u_prev_class_tex", "u_feedback", "u_use_iso_sigma", "u_iso_model_a")) {
            assertTrue("$gone was a Phase-2 binding and must not come back", !map.contains(gone))
        }
    }

    /**
     * Locating a uniform is not uploading it.
     *
     * This is the specific false negative that produced a device-only ratio
     * blow-up worth two orders of magnitude. A `glGetUniformLocation` call with
     * no matching `glUniform1f` compiles, links, and runs: the sampler gets its
     * texture, the shader reads the uniform's default of 0.0, and the ratio
     * divides by a floored epsilon instead of the noise prediction. Nothing on
     * the device reports it, and a suite that only asserts the location exists
     * passes it happily.
     *
     * So each fold uniform is pinned through all three stages: declared in the
     * GLSL, located on the host, and written inside draw() through the location
     * field that draw() uploads from. The mapping is spelled out rather than
     * derived, so renaming a field fails here instead of silently unpairing it.
     */
    @Test
    fun everyFoldUniformIsUploadedByDrawNotJustLocated() {
        val map = source("StructureMapShaderProgram")
        val glsl = rawString(map, "FRAGMENT_SHADER")
        val draw = map
            .substringAfter("fun draw(")
            .substringBefore("fun isReady")

        // GLSL uniform name -> the host location field draw() uploads through.
        val bindings = listOf(
            "u_luma_eps_scale" to "uLumaEpsScaleLoc",
            "u_sigma_dm2" to "uSigmaDm2Loc",
            "u_inverse_range2" to "uInverseRange2Loc",
            "u_calib_scale" to "uCalibScaleLoc",
            "u_eps_boost" to "uEpsBoostLoc"
        )
        for ((uniform, field) in bindings) {
            assertTrue("$uniform must be declared in the GLSL", glsl.contains("uniform float $uniform;"))
            assertTrue(
                "$uniform must be located on the host",
                map.contains("""glGetUniformLocation(programId, "$uniform")""")
            )
            assertTrue(
                "the location field for $uniform ($field) must be declared",
                map.contains("private var $field = 0")
            )
            assertTrue(
                "$uniform is located but never uploaded: draw() has no " +
                    "glUniform1f($field, ...). A located-but-unwritten uniform " +
                    "silently reads as 0.0 on the device.",
                draw.contains("glUniform1f($field,")
            )
        }
    }
}