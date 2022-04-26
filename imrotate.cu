/*__global__ void Rotate3D(float* Destination,  float* Source, int sizeX, int sizeY, float deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;// Kernel definition
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int xc = sizeX - sizeX/2;
    int yc = sizeY - sizeY/2;
    int newx = ((float)i-xc)*cos(deg) + ((float)j-yc)*sin(deg) + xc;
    int newy = -((float)i-xc)*sin(deg) + ((float)j-yc)*cos(deg) + yc;
    if (newx >= 0 && newx < sizeX && newy >= 0 && newy < sizeY)
    {
        // putPixVal(Destination, sizeX, i , j, readPixVal(Source, sizeX, newx, newy));
	Destination[k* sizeX*sizeY + j*sizeX+i] = Source[k* sizeX*sizeY + newy*sizeX + newx];
    }
} */

// rotate a 3D array (2D rotate, with the third dim simultaenously processed) with bilinear interpolation
# define pi 3.141592654f
__global__ void Rotate3D_bilinear(float* Destination,  float* Source, int sizeX, int sizeY, float deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;// Kernel definition
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int z_offset = k*sizeX*sizeY;
    int xc = sizeX - sizeX/2;
    int yc = sizeY - sizeY/2;
    float cos_rad = cosf(deg/180*pi);
    float sin_rad = sinf(deg/180*pi);
    float newx = +((float)i-xc)*cos_rad  + ((float)j-yc)*sin_rad + xc ;
    float newy = -((float)i-xc)*sin_rad + ((float)j-yc)*cos_rad  + yc ;
    int newx_0 = floorf(newx);
    int newy_0 = floorf(newy);
    int newx_1 = newx_0 + 1;
    int newy_1 = newy_0 + 1;
    // put the pixel coo. in unit square
    float x = newx - float(newx_0);
    float y = newy - float(newy_0);
    float val_00 = 0.0, val_01 = 0.0,  val_10 = 0.0, val_11 = 0.0, val_intp = 0.0;
    if (newx_0 >= 0 && newx_1 < sizeX && newy_0 >= 0 && newy_1 < sizeY)
    {
        // perform bilinear interpolation
	val_00 = Source[z_offset + newy_0*sizeX + newx_0];
	val_01 = Source[z_offset + newy_1*sizeX + newx_0];
	val_10 = Source[z_offset + newy_0*sizeX + newx_1];
	val_11 = Source[z_offset + newy_1*sizeX + newx_1];
	val_intp = val_00 + (val_01-val_00)*y + (val_10-val_00)*x + (val_11 + val_00 - val_10 - val_01)*x*y;
	Destination[z_offset + j*sizeX+i] = val_intp;
    }
}