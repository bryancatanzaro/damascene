__global__ void
parabolaKernel(float* trace, int width, int height, int limit, int border, int border_height, int filter_radius, int filter_length, int filter_width)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int ay = blockIdx.y*blockDim.y + threadIdx.y;

    if (x < width && ay < limit)
    {
        int ori = ay / height;
        int y = ay % height;

        float val = 0;

        for (int v=-filter_radius; v<=filter_radius; v++)
        {
            int cy = y + border + v + ori*border_height;
            int fidx = (v+filter_radius)*filter_length+ori*filter_width;

            for (int u=-filter_radius; u<=filter_radius; u++)
            {
                int cx = x + border + u;
                val += (tex2D(tex_parabola_pixels, cx, cy) * const_parabola_filters[fidx+u+filter_radius]);
                //val += (tex2D(tex_parabola_pixels, cx, cy) * tex1Dfetch(tex_parabola_filters, fidx+u+filter_radius));//[fidx+u+filter_radius]);
            }
        }

        trace[x+y*width+ori*width*height] = val;
        //trace[x+y*width+ori*width*height] = tex2D(tex_parabola_pixels, x+border, y+border);
    }
}
