#version 430
//#extension GL_KHR_storage_buffer_storage_class : enable

layout(binding = 0) uniform UniformBufferObject {
    int width;
    int height;

    int in_channels;
    int out_channels;
    int kernel_width;
    int kernel_height;
    int kernel_stride_x;
    int kernel_stride_y;
//actually padding the array will be done elsewhere, or not at all. This indicates which areas to ignore.
//global x and y size should be smaller to deal with restricted area
    int kernel_padding_x;
    int kernel_padding_y;
    int kernel_dilation_x;
    int kernel_dilation_y;
} c;

layout (std430, binding = 0) buffer InputImage {
    float in_pixels[];
};

layout (std430, binding = 1) buffer InputKernal {
    float in_kernel[];
};

#ifdef KERNEL_BIAS
layout (std430, binding = 3) buffer KernelBias {
    float out_bias[];
};
#endif

layout (std430, binding = 2) buffer OutputImage {
    float out_pixels[];
};

int get_input_pixel(int input_sel, int from_x, int from_y){
    return input_sel + from_x*c.width*c.in_channels + from_y*c.in_channels;
}

layout (local_size_x = 1, local_size_y = 1) in;
void main() {
    ivec2 ourPos = ivec2(gl_GlobalInvocationID.xy);
    int output_select = (ourPos.x*c.width*c.out_channels)+ourPos.y*c.out_channels;
    int kernel_select;
    int i_f = int(floor(c.kernel_width/2.0));
    int j_f = int(floor(c.kernel_height/2.0));
    int input_select = ((ourPos.x*c.kernel_stride_x+c.kernel_padding_x+i_f)*c.width*c.in_channels)+(ourPos.y*c.kernel_stride_y+c.kernel_padding_y+j_f)*c.in_channels;
    int in_pixel_select;

    for (int i=-int(floor(c.kernel_width/2.0)); i<floor(c.kernel_width/2.0); i++){
        for (int j=-int(floor(c.kernel_height/2.0)); j<floor(c.kernel_height/2.0); j++){
            in_pixel_select = get_input_pixel(input_select, i, j);
            for (int co=0;co<c.out_channels; co++){
                for (int ci=0; ci<c.in_channels; ci++){
                    kernel_select = (i_f+i)*c.kernel_height*c.out_channels*c.in_channels+(j_f+j)*c.out_channels*c.in_channels+ci*c.out_channels+co;
                    out_pixels[output_select+co] += in_kernel[kernel_select]*in_pixels[in_pixel_select+ci];
                }
                #ifdef KERNEL_BIAS
                out_pixels[output_select+co] += out_bias[output_select+co];
                #endif
            }
        }
    }

}
