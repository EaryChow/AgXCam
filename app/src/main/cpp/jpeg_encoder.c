#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <android/log.h>
#include <jpeglib.h>

#define LOG_TAG "JpegEncoderNative"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

typedef struct {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
} my_error_mgr;

typedef my_error_handler * my_error_ptr;

static void my_error_exit(j_common_ptr cinfo) {
    my_error_ptr err = (my_error_ptr)cinfo->err;
    longjmp(err->setjmp_buffer, 1);
}

typedef struct {
    unsigned char *data;
    size_t size;
    size_t capacity;
} mem_destination_mgr;

static void init_mem_destination(j_compress_ptr cinfo) {
    mem_destination_mgr *dest = (mem_destination_mgr *)cinfo->dest;
    dest->size = 0;
}

static boolean empty_mem_buffer(j_compress_ptr cinfo) {
    mem_destination_mgr *dest = (mem_destination_mgr *)cinfo->dest;
    size_t new_capacity = dest->capacity * 2;
    unsigned char *new_data = (unsigned char *)realloc(dest->data, new_capacity);
    if (!new_data) return FALSE;
    dest->data = new_data;
    dest->capacity = new_capacity;
    return TRUE;
}

static void term_mem_destination(j_compress_ptr cinfo) {}

JNIEXPORT jbyteArray JNICALL
Java_com_agx_camera_io_JpegEncoder_nativeEncodeRgba(
        JNIEnv *env, jclass clazz,
        jobject rgbaBuffer, jint width, jint height, jint quality) {

    unsigned char *rgba = (unsigned char *)(*env)->GetDirectBufferAddress(env, rgbaBuffer);
    if (!rgba) {
        LOGE("Failed to get direct buffer address");
        return NULL;
    }

    struct jpeg_compress_struct cinfo;
    my_error_mgr jerr;

    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_compress(&cinfo);
        return NULL;
    }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jpeg_create_compress(&cinfo);

    mem_destination_mgr dest;
    dest.data = (unsigned char *)malloc(width * height * 3);
    if (!dest.data) {
        LOGE("Failed to allocate destination buffer");
        jpeg_destroy_compress(&cinfo);
        return NULL;
    }
    dest.capacity = width * height * 3;
    dest.size = 0;
    dest.pub.init_destination = init_mem_destination;
    dest.pub.empty_output_buffer = empty_mem_buffer;
    dest.pub.term_destination = term_mem_destination;
    cinfo.dest = &dest;

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    cinfo.dct_method = JDCT_FASTEST;

    jpeg_start_compress(&cinfo, TRUE);

    unsigned char *row = (unsigned char *)malloc(width * 3);
    if (!row) {
        LOGE("Failed to allocate row buffer");
        jpeg_destroy_compress(&cinfo);
        free(dest.data);
        return NULL;
    }
    while (cinfo.next_scanline < cinfo.image_height) {
        unsigned char *src = rgba + cinfo.next_scanline * width * 4;
        for (int x = 0; x < width; x++) {
            row[x * 3 + 0] = src[x * 4 + 0];
            row[x * 3 + 1] = src[x * 4 + 1];
            row[x * 3 + 2] = src[x * 4 + 2];
        }
        jpeg_write_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    free(row);

    jbyteArray result = (*env)->NewByteArray(env, (jsize)dest.size);
    if (result) {
        (*env)->SetByteArrayRegion(env, result, 0, (jsize)dest.size, (jbyte *)dest.data);
    }
    free(dest.data);

    return result;
}
