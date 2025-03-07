#ifdef __ARM_NEON
#define STBI_NEON
#endif

#include <stdbool.h>
#include "rknn_api.h"
#include "yolov5.h"

int main(int argc, char** argv){
    // get img path
    const char* img_path;
    if (argc > 1){
        img_path = argv[1];
    } else {
        img_path = "dog.jpg";
    }

    // read image
    int width, height, channels;
    unsigned char *img = stbi_load(img_path, &width, &height, &channels, 0);
    if (img == NULL) {
            printf("Error in loading the image\n");
            exit(1);
    }
    printf("img: %s, width = %d, height = %d, channels = %d\n", img_path, width, height, channels);

    // init context
    rknn_context ctx;
    rknn_init(&ctx, "yolov5s_v6.1_int8_1b.rknn", 0, 0, NULL);
    rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO);

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr attr;
    attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));

    // init r_info
    struct resize_info r_info;
    r_info.ori_w = width;
    r_info.ori_h = height;
    r_info.net_w = attr.dims[1];
    r_info.net_h = attr.dims[2];
    r_info.ratio_x = (float)r_info.net_w/r_info.ori_w;
    r_info.ratio_y = (float)r_info.net_h/r_info.ori_h;
    r_info.start_x = 0;
    r_info.start_y = 0;
    r_info.keep_aspect = false;

    // prepare input data memory
    uint8_t* input_data = (uint8_t*)malloc(r_info.net_w*r_info.net_h*3);
    stbir_resize_uint8_linear(img, width, height, 0, input_data, r_info.net_w, r_info.net_h, 0, STBIR_RGB);

    // input
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = r_info.net_w*r_info.net_h*attr.dims[3];
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input_data;
    inputs[0].pass_through = 0;
    rknn_inputs_set(ctx, 1, inputs);

    // inference
    rknn_run(ctx, NULL);

    // prepare output
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].is_prealloc = 0;
        outputs[i].want_float = 1;
    }
    rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    float* output[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++) {
	output[i] = outputs[i].buf;
    }

    // do postprocess
    post_process(output, img_path, img, &r_info);

    stbi_image_free(img);

    free(input_data);
    rknn_outputs_release(ctx, io_num.n_output, outputs);
    rknn_destroy(ctx);

    return 0;
}
