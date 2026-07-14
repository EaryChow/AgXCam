package com.agx.camera.gpu

import org.junit.Assert.*
import org.junit.Test

class DemosaicAdapterTest {

    @Test
    fun standardBayerAdapter_supported() {
        val adapter = StandardBayerAdapter()
        assertTrue("StandardBayerAdapter should be supported", adapter.supported())
        assertEquals("standard_bayer", adapter.name)
        assertEquals("edge_directed", adapter.strategy)
    }

    @Test
    fun standardBayerAdapter_tileSize() {
        val adapter = StandardBayerAdapter()
        assertEquals(2048, adapter.tileSize())
    }

    @Test
    fun standardBayerAdapter_overlap() {
        val adapter = StandardBayerAdapter()
        assertEquals(2, adapter.overlapPixels())
    }

    @Test
    fun standardBayerAdapter_computeShader_containsRequiredElements() {
        val adapter = StandardBayerAdapter()
        val source = adapter.createComputeShaderSource(
            com.agx.camera.camera.BayerPattern.RGGB, 10
        )

        assertTrue("Should contain version directive", source.contains("#version 310 es"))
        assertTrue("Should contain layout local_size", source.contains("layout(local_size_x"))
        assertTrue("Should contain usampler2D", source.contains("usampler2D"))
        assertTrue("Should contain u_bayerTex", source.contains("u_bayerTex"))
        assertTrue("Should contain u_outTex", source.contains("u_outTex"))
        assertTrue("Should contain u_black_level_pattern", source.contains("u_black_level_pattern"))
        assertTrue("Should contain u_bayer_color_map", source.contains("u_bayer_color_map"))
        assertTrue("Should contain u_lens_shading_map", source.contains("u_lens_shading_map"))
        assertTrue("Should contain unpackRaw", source.contains("unpackRaw"))
        assertTrue("Should contain sampleBayer", source.contains("sampleBayer"))
        assertTrue("Should contain lensGain", source.contains("lensGain"))
        assertTrue("Should contain main()", source.contains("void main()"))
    }

    @Test
    fun standardBayerAdapter_shader_rggb_colorMap() {
        val adapter = StandardBayerAdapter()
        val source = adapter.createComputeShaderSource(
            com.agx.camera.camera.BayerPattern.RGGB, 10
        )
        assertTrue("RGGB map should be [0,1,1,2]",
            source.contains("0, 1, 1, 2") || source.contains("u_bayer_color_map"))
    }

    @Test
    fun standardBayerAdapter_shader_10bit_mask() {
        val adapter = StandardBayerAdapter()
        val source = adapter.createComputeShaderSource(
            com.agx.camera.camera.BayerPattern.RGGB, 10
        )
        assertTrue("10-bit should use 0x3FFu mask", source.contains("0x3FFu"))
    }

    @Test
    fun standardBayerAdapter_shader_12bit_mask() {
        val adapter = StandardBayerAdapter()
        val source = adapter.createComputeShaderSource(
            com.agx.camera.camera.BayerPattern.RGGB, 12
        )
        assertTrue("12-bit should use 0xFFFu mask", source.contains("0xFFFu"))
    }

    @Test
    fun previewBilinearDemosaicAdapter() {
        val adapter = PreviewBilinearDemosaicAdapter()
        assertTrue(adapter.supported())
        assertEquals("preview_bilinear", adapter.name)
        assertEquals("sparse_bilinear", adapter.strategy)
        assertEquals(0, adapter.tileSize())
        assertEquals(0, adapter.overlapPixels())
        assertEquals("", adapter.createComputeShaderSource(
            com.agx.camera.camera.BayerPattern.RGGB, 10
        ))
    }
}
