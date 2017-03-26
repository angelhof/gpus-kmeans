#include <stdio.h> 
#include <cuda.h> 


int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if (devProp.minor == 1) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

void show_device_info(cudaDeviceProp prop, int cnt){
  printf("Device Number: %d\n", cnt);
  printf("  Device name: %s\n", prop.name);
  printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("  Max Block Dimensions: %d %d %d\n", prop.maxThreadsDim[0],
												 prop.maxThreadsDim[1],
												 prop.maxThreadsDim[2]);
  printf("  Max Grid Dimensions: %d %d %d\n", prop.maxGridSize[0],
												 prop.maxGridSize[1],
												 prop.maxGridSize[2]);
  printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
         2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  return;
}


int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    show_device_info(prop, i);

    int device_cores, device_mps;

    device_cores = getSPcores(prop);
    device_mps = prop.multiProcessorCount;
    printf(" Device Cores: %d\n", device_cores);
    printf(" Device Multi Processors: %d\n", device_mps);
  }

}


