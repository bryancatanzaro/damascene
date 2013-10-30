// vim:ts=4 syntax=c
#include <xmmintrin.h>
#include <omp.h>

void
cpu_convolve_xmm (
	float *_in, int const in_pitch, float *_out, int const out_pitch,
	int border_x, int border_y,
	int const width, int const height, int const radius, int const ori,
	float const *filters, int const filter_size, int const filter_pitch)
{
	int x, y, dx, dy;

	float const * __restrict__ const in = _in + border_x + border_y * in_pitch;
	float const * __restrict__ const f = filters + ori*filter_size
	                                     + radius + radius * filter_pitch;
	float * __restrict__ const out = _out + border_x + border_y * out_pitch;


#pragma omp parallel for schedule(static) private(x,y,dx,dy)
	for (y = 0; y < height; y += 2)
	{
		if (y+2 <= height)
		{
			for (x = 0; x+16 <= width; x += 16)
			{
				__m128 v00 = _mm_setzero_ps();
				__m128 v10 = _mm_setzero_ps();
				__m128 v20 = _mm_setzero_ps();
				__m128 v30 = _mm_setzero_ps();

				__m128 v01 = _mm_setzero_ps();
				__m128 v11 = _mm_setzero_ps();
				__m128 v21 = _mm_setzero_ps();
				__m128 v31 = _mm_setzero_ps();

				for (dy = -radius; dy <= radius; dy++){
					for (dx = -radius; dx <= radius; dx++)
					{
						__m128 f_dx_dy;

						__m128 in_x0dx_y0dy;
						__m128 in_x1dx_y0dy;
						__m128 in_x2dx_y0dy;
						__m128 in_x3dx_y0dy;

						__m128 in_x0dx_y1dy;
						__m128 in_x1dx_y1dy;
						__m128 in_x2dx_y1dy;
						__m128 in_x3dx_y1dy;

						f_dx_dy = _mm_load1_ps (&f[dx + filter_pitch*dy]);

						in_x0dx_y0dy = _mm_loadu_ps(&in[x+dx+in_pitch*(y+dy)]);
						in_x1dx_y0dy = _mm_loadu_ps(&in[x+dx+4+in_pitch*(y+dy)]);
						in_x2dx_y0dy = _mm_loadu_ps(&in[x+dx+8+in_pitch*(y+dy)]);
						in_x3dx_y0dy = _mm_loadu_ps(&in[x+dx+12+in_pitch*(y+dy)]);

						in_x0dx_y1dy = _mm_loadu_ps(&in[x+dx   +in_pitch*(y+1+dy)]);
						in_x1dx_y1dy = _mm_loadu_ps(&in[x+dx+4 +in_pitch*(y+1+dy)]);
						in_x2dx_y1dy = _mm_loadu_ps(&in[x+dx+8 +in_pitch*(y+1+dy)]);
						in_x3dx_y1dy = _mm_loadu_ps(&in[x+dx+12+in_pitch*(y+1+dy)]);

						v00 = _mm_add_ps(v00, _mm_mul_ps (f_dx_dy, in_x0dx_y0dy));
						v10 = _mm_add_ps(v10, _mm_mul_ps (f_dx_dy, in_x1dx_y0dy));
						v20 = _mm_add_ps(v20, _mm_mul_ps (f_dx_dy, in_x2dx_y0dy));
						v30 = _mm_add_ps(v30, _mm_mul_ps (f_dx_dy, in_x3dx_y0dy));

						v01 = _mm_add_ps(v01, _mm_mul_ps (f_dx_dy, in_x0dx_y1dy));
						v11 = _mm_add_ps(v11, _mm_mul_ps (f_dx_dy, in_x1dx_y1dy));
						v21 = _mm_add_ps(v21, _mm_mul_ps (f_dx_dy, in_x2dx_y1dy));
						v31 = _mm_add_ps(v31, _mm_mul_ps (f_dx_dy, in_x3dx_y1dy));
					}
				}

				_mm_storeu_ps(&out[x    + out_pitch*y], v00);
				_mm_storeu_ps(&out[x+4  + out_pitch*y], v10);
				_mm_storeu_ps(&out[x+8  + out_pitch*y], v20);
				_mm_storeu_ps(&out[x+12 + out_pitch*y], v30);

				_mm_storeu_ps(&out[x    + out_pitch*(y+1)], v01);
				_mm_storeu_ps(&out[x+4  + out_pitch*(y+1)], v11);
				_mm_storeu_ps(&out[x+8  + out_pitch*(y+1)], v21);
				_mm_storeu_ps(&out[x+12 + out_pitch*(y+1)], v31);
			}

			for (; x < width; x++)
			{
				float v0 = 0.0;
				float v1 = 0.0;

				for (dy = -radius; dy <= radius; dy++){
					for (dx = -radius; dx <= radius; dx++)
					{
						v0 += in[x+dx + in_pitch*(y+dy)] * f[dx + filter_pitch*dy];
						v1 += in[x+dx + in_pitch*(y+1+dy)] * f[dx + filter_pitch*dy];
					}
				}

				out[x + out_pitch*y] = v0;
				out[x + out_pitch*(y+1)] = v1;
			}

		}else{

			for (x = 0; x+16 <= width; x += 16)
			{
				__m128 v0 = _mm_setzero_ps();
				__m128 v1 = _mm_setzero_ps();
				__m128 v2 = _mm_setzero_ps();
				__m128 v3 = _mm_setzero_ps();

				for (dy = -radius; dy <= radius; dy++){
					for (dx = -radius; dx <= radius; dx++)
					{
						__m128 f_dx_dy;

						__m128 in_x0dx_ydy;
						__m128 in_x1dx_ydy;
						__m128 in_x2dx_ydy;
						__m128 in_x3dx_ydy;

						f_dx_dy = _mm_load1_ps (&f[dx + filter_pitch*dy]);

						in_x0dx_ydy = _mm_loadu_ps (&in[x+dx    + in_pitch*(y+dy)]);
						in_x1dx_ydy = _mm_loadu_ps (&in[x+dx+4  + in_pitch*(y+dy)]);
						in_x2dx_ydy = _mm_loadu_ps (&in[x+dx+8  + in_pitch*(y+dy)]);
						in_x3dx_ydy = _mm_loadu_ps (&in[x+dx+12 + in_pitch*(y+dy)]);

						v0 = _mm_add_ps(v0, _mm_mul_ps (f_dx_dy, in_x0dx_ydy));
						v1 = _mm_add_ps(v1, _mm_mul_ps (f_dx_dy, in_x1dx_ydy));
						v2 = _mm_add_ps(v2, _mm_mul_ps (f_dx_dy, in_x2dx_ydy));
						v3 = _mm_add_ps(v3, _mm_mul_ps (f_dx_dy, in_x3dx_ydy));
					}
				}

				_mm_storeu_ps(&out[x    + out_pitch*y], v0);
				_mm_storeu_ps(&out[x+4  + out_pitch*y], v1);
				_mm_storeu_ps(&out[x+8  + out_pitch*y], v2);
				_mm_storeu_ps(&out[x+12 + out_pitch*y], v3);
			}

			for (; x < width; x++)
			{
				float v = 0.0;

				for (dy = -radius; dy <= radius; dy++){
					for (dx = -radius; dx <= radius; dx++){
						v += in[x+dx + in_pitch*(y+dy)] * f[dx + filter_pitch*dy];
					}
				}
				out[x + out_pitch*y] = v;
			}
		}
	}
}

