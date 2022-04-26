// ****
// Implement the phasor method using complex data rather than two times reconstruction
// ****
#define NPOINT 1
#define STRIDE 1
/*__global__ void ThreeD_NLOS_Phasor_General( float* p_xyz,  float* p_xyt_real,  float* p_xyt_imag,   
    float* sensor_pos, float* origin, float* laser_pos,  float dx,  float dz,  int NX,  int NY, int NZ, 
    float c_dt,  int N_pos, int Nt,  float ReceiveDelay)
{
        int nz = blockIdx.z * blockDim.z + threadIdx.z; // the nz-column, which is the time direction
        int ny = blockIdx.y * blockDim.y + threadIdx.y; // the ny-col
        int nx = blockIdx.x * blockDim.x + threadIdx.x; // the nx-row
        float CF_real = 0.0, CF_imag = 0.0, CF = 0.0;
        float tao = 0.0;
        float voxel_int_real = 0.0, voxel_int_imag =0.0;
        float voxel_real = 0.0, voxel_imag =0.0;
        int Index = 0, counter = 1;
        float GPS_x = origin[0] - (nx-NX/2)*dx;    // x-coordiante for the voxel
        float GPS_y = origin[1] + (ny-NY/2)*dx;    // y-coordiante for the voxel
        float GPS_z = origin[2] + (nz)*dz;         // z-coordiante for the voxel
        // time delay from the laser spot to the voxel
        float tao_trans = sqrtf( (laser_pos[0]-GPS_x)*(laser_pos[0]-GPS_x) + (laser_pos[1]-GPS_y)*(laser_pos[1]-GPS_y) + (laser_pos[2]-GPS_z)*(laser_pos[2]-GPS_z) );
        float x_pos,y_pos,z_pos, cos_theta;
        float voxel_int_real_tp[NPOINT], voxel_int_imag_tp[NPOINT],  voxel_sq_real_tp[NPOINT], voxel_sq_imag_tp[NPOINT];
        
        for(int K = 0; K<NPOINT; K++)
        {
            voxel_int_real_tp[K] = 0.0;
            voxel_int_imag_tp[K] = 0.0;
            voxel_sq_real_tp[K] = 0.0;
            voxel_sq_imag_tp[K] = 0.0;
        }
        
        for(int K_x = 0;K_x<N_pos; K_x++)
        {
              x_pos = sensor_pos[K_x];
              y_pos = sensor_pos[K_x+N_pos];
              z_pos = sensor_pos[K_x+2*N_pos];
              tao = sqrtf( (x_pos-GPS_x) * (x_pos-GPS_x) + (y_pos-GPS_y)*(y_pos-GPS_y) + (z_pos-GPS_z)*(z_pos-GPS_z) );
              cos_theta = GPS_z/(tao + 1e-6);
              Index = int( floorf((tao + tao_trans)/c_dt-ReceiveDelay) );
              if((Index<Nt-NPOINT*STRIDE) && (Index>0))
              {
                  for(int P = 0; P<NPOINT; P++)
                    {
                        voxel_real = p_xyt_real[Index+K_x*(Nt)+P*STRIDE] * cos_theta;
                        voxel_imag = p_xyt_imag[Index+K_x*(Nt)+P*STRIDE] * cos_theta;                         
                        voxel_int_real_tp[P] = voxel_int_real_tp[P] + voxel_real;    //* sqrt(Index*1.0);
                        voxel_int_imag_tp[P] = voxel_int_imag_tp[P] + voxel_imag;    //* sqrt(Index*1.0);
                        voxel_sq_real_tp[P] = voxel_sq_real_tp[P] + voxel_real * voxel_real;
                        voxel_sq_imag_tp[P] = voxel_sq_imag_tp[P] + voxel_imag * voxel_imag;
                    }
                    counter = counter+1;
              }
        }
        
        voxel_int_real = voxel_int_real_tp[0]; 
        voxel_int_imag = voxel_int_imag_tp[0];
        
        
        for (int J=0;J<NPOINT;J++)
        {
    
            if((voxel_sq_real_tp[J]>1e-6) && (voxel_sq_imag_tp[J]>1e-6))
            {
              CF_real = CF_real + powf(voxel_int_real_tp[J],2)/voxel_sq_real_tp[J]/counter;
              CF_imag = CF_imag + powf(voxel_int_imag_tp[J],2)/voxel_sq_imag_tp[J]/counter;
              
            }
        }
        CF = sqrtf( CF_real*CF_real + CF_imag*CF_imag );
        p_xyz[nx+NX*ny+nz*(NX*NY)] = sqrtf( voxel_int_real * voxel_int_real  + voxel_int_imag * voxel_int_imag ) * (CF);
}*/

// ****
// Implement the phasor method using complex data rather than two times reconstruction
// ****
#define NPOINT 1
#define STRIDE 1
__global__ void ThreeD_NLOS_Phasor_General( float* p_xyz,  float* p_xyt_real,  float* p_xyt_imag,   
    float* sensor_pos, float* origin, float* laser_pos,  float dx,  float dz,  int NX,  int NY, int NZ, 
    float c_dt,  int N_pos, int Nt,  float ReceiveDelay)
{
        int nz = blockIdx.z * blockDim.z + threadIdx.z; // the nz-column, which is the time direction
        int ny = blockIdx.y * blockDim.y + threadIdx.y; // the ny-col
        int nx = blockIdx.x * blockDim.x + threadIdx.x; // the nx-row
        float CF_real = 0.0, CF_imag = 0.0, CF = 0.0;
        float tao = 0.0;
        float voxel_int_real = 0.0, voxel_int_imag =0.0;
        float voxel_real = 0.0, voxel_imag =0.0;
        int Index = 0, counter = 1;
        float GPS_x = origin[0] - (nx-NX/2)*dx;    // x-coordiante for the voxel
        float GPS_y = origin[1] + (ny-NY/2)*dx;    // y-coordiante for the voxel
        float GPS_z = origin[2] + (nz)*dz;         // z-coordiante for the voxel
        // time delay from the laser spot to the voxel
        float tao_trans = sqrtf( (laser_pos[0]-GPS_x)*(laser_pos[0]-GPS_x) + (laser_pos[1]-GPS_y)*(laser_pos[1]-GPS_y) + (laser_pos[2]-GPS_z)*(laser_pos[2]-GPS_z) );
        float x_pos,y_pos,z_pos, cos_theta;
        float  voxel_sq_real, voxel_sq_imag;
         
        for(int K_x = 0;K_x<N_pos; K_x++)
        {
              x_pos = sensor_pos[K_x];
              y_pos = sensor_pos[K_x+N_pos];
              z_pos = sensor_pos[K_x+2*N_pos];
              tao = sqrtf( (x_pos-GPS_x) * (x_pos-GPS_x) + (y_pos-GPS_y)*(y_pos-GPS_y) + (z_pos-GPS_z)*(z_pos-GPS_z) );
              // cos_theta = GPS_z/(tao + 1e-6);
              Index = int( floorf((tao + tao_trans)/c_dt-ReceiveDelay) );
              if((Index<Nt) && (Index>0))
              {

                    voxel_real = p_xyt_real[Index+K_x*(Nt)]; // * cos_theta;
                    voxel_imag = p_xyt_imag[Index+K_x*(Nt)]; // * cos_theta;                         
                    voxel_int_real = voxel_int_real + voxel_real;    //* sqrt(Index*1.0);
                    voxel_int_imag = voxel_int_imag + voxel_imag;    //* sqrt(Index*1.0);
                    voxel_sq_real = voxel_sq_real + voxel_real * voxel_real;
                    voxel_sq_imag = voxel_sq_imag + voxel_imag * voxel_imag;
                    counter = counter+1;
              }
        }
              
        if((voxel_sq_real>1e-6) && (voxel_sq_imag>1e-6))
        {
          CF_real = CF_real + powf(voxel_int_real,2)/voxel_sq_real/counter;
          CF_imag = CF_imag + powf(voxel_int_imag,2)/voxel_sq_imag/counter;
          
        }
 
        CF = sqrtf( CF_real*CF_real + CF_imag*CF_imag );
        p_xyz[nx+NX*ny+nz*(NX*NY)] = sqrtf( voxel_int_real * voxel_int_real  + voxel_int_imag * voxel_int_imag ) * (CF);
}