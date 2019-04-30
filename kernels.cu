#ifndef KERNELS_H_
#define KERNELS_H_
#define K_DIM_LIMIT 1000
#include <float.h>
__constant__ double centroidCONST[K_DIM_LIMIT];

//#include "sm_60_atomic_functions.h"

//#if __CUDA_ARCH__ < 600
//AWS g3x.large computecapability is 5.2, atomic doubles implemeneted in 6 :(
__device__ double myAtomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  
  return __longlong_as_double(old);
}
//#endif


//Each thread chooses one vector and calcs distance to every centroid
__global__ void vectorDistance(double * data, double *  centroids, double *  distances, int k, int dim, int n){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension++){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
      }
      distance = sqrt(distance);
      distances[(vector*k)+centroid] = distance;
    }
  }
}

//Each thread chooses one vector and calcs distance to every centroid
__global__ void vectorDistance1(double * data, double *  centroids, double *  distances, int k, int dim, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if( vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension++){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp*temp;
      }
      distances[(vector*k)+centroid] = sqrt(distance);
    }
  }
}

__global__ void vectorDistance2(const double * __restrict__ data, const double * __restrict__ centroids, double *  __restrict__ distances, int k, int dim, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if( vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension++){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp*temp;
      }
      distances[(vector*k)+centroid] = sqrt(distance);
    }
  }
}

__global__ void vectorDistance2UR2(const double * __restrict__ data, const double * __restrict__ centroids, double *  __restrict__ distances, int k, int dim, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if( vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension+=2){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
	distance += temp*temp;
      }
      distances[(vector*k)+centroid] = distance = sqrt(distance);
    }
  }
}

__global__ void vectorDistance2UR4(const double * __restrict__ data, const double * __restrict__ centroids, double *  __restrict__ distances, int k, int dim, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if( vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension += 4){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+2] - data[vector*dim+dimension+2];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+3] - data[vector*dim+dimension+3];
	distance += temp*temp;
      }
      distances[(vector*k)+centroid] = distance = sqrt(distance);
    }
  }
}

__global__ void vectorDistance2UR8(const double * __restrict__ data, const double * __restrict__ centroids, double *  __restrict__ distances, int k, int dim, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if( vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension+=8){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+2] - data[vector*dim+dimension+2];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+3] - data[vector*dim+dimension+3];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+4] - data[vector*dim+dimension+4];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+5] - data[vector*dim+dimension+5];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+6] - data[vector*dim+dimension+6];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+7] - data[vector*dim+dimension+7];
	distance += temp*temp;
      }
      distances[(vector*k)+centroid] = distance = sqrt(distance);
    }
  }
}

//Each thread chooses one vector and then one centroid and calcs that distance
//More ocupancy
__global__ void centroidDistance(double * data, double *  centroids, double *  distances, int k, int dim, int centroid, int n){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    double distance = 0;
    for(int dimension = 0; dimension < dim; dimension++){
      double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
      distance += temp*temp;
    }
    distance = sqrt(distance);
    distances[(vector*k)+centroid] = distance;
  }
}

__global__ void centroidDistance1(double * data, double *  centroids, double *  distances, int k, int dim, int centroid){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  double distance = 0;
  for(int dimension = 0; dimension < dim; dimension++){
    double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
    distance += temp*temp;
  }
  distances[(vector*k)+centroid] = sqrt(distance);;
}

__global__ void centroidDistance2(const double * __restrict__ data,const double * __restrict__ centroids, double * __restrict__  distances, int k, int dim, int centroid, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector >= n) return;
  double distance = 0;
  for(int dimension = 0; dimension < dim; dimension++){
    double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
    distance += temp*temp;
  }
  distances[(vector*k)+centroid] = sqrt(distance);;
}

__global__ void centroidDistance2UR2(const double * __restrict__ data,const double * __restrict__ centroids, double * __restrict__  distances, int k, int dim, int centroid, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector >= n) return;
  double distance = 0;
  for(int dimension = 0; dimension < dim; dimension+=2){
    double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
    distance += temp*temp;
  }
  distances[(vector*k)+centroid] = sqrt(distance);;
}

__global__ void centroidDistance2UR4(const double * __restrict__ data,const double * __restrict__ centroids, double * __restrict__  distances, int k, int dim, int centroid, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector >= n) return;
  double distance = 0;
  for(int dimension = 0; dimension < dim; dimension+=4){
    double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+2] - data[vector*dim+dimension+2];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+3] - data[vector*dim+dimension+3];
    distance += temp*temp;
  }
  distances[(vector*k)+centroid] = sqrt(distance);;
}

__global__ void centroidDistance2UR8(const double * __restrict__ data,const double * __restrict__ centroids, double * __restrict__  distances, int k, int dim, int centroid, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector >= n) return;
  double distance = 0;
  for(int dimension = 0; dimension < dim; dimension+=8){
    double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+2] - data[vector*dim+dimension+2];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+3] - data[vector*dim+dimension+3];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+4] - data[vector*dim+dimension+4];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+5] - data[vector*dim+dimension+5];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+6] - data[vector*dim+dimension+6];
    distance += temp*temp;
    temp = centroids[centroid*dim+dimension+7] - data[vector*dim+dimension+7];
    distance += temp*temp;
  }
  distances[(vector*k)+centroid] = sqrt(distance);;
}

//Each thread chooses
__global__ void centroidConstantDistance(double * data,  double *  distances, int k, int dim, int centroid, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector < n){
    double distance = 0;
    for(int dimension = 0; dimension < dim; dimension++){
      double temp = centroidCONST[dimension] - data[vector*dim+dimension];
      distance += temp*temp;
    }
    distance = sqrt(distance);
    distances[(vector*k)+centroid] = distance;
  }
}

__global__ void centroidConstantDistance1(double * data,  double *  distances, int k, int dim, int centroid){
  long  vector = blockIdx.x * blockDim.x +threadIdx.x;
  double distance = 0;
  for(int dimension = 0; dimension < dim; dimension++){
    double temp = centroidCONST[dimension] - data[vector*dim+dimension];
    distance += temp*temp;
  }
  distances[(vector*k)+centroid] = sqrt(distance);
}

__global__ void centroidConstantDistance2(const double * __restrict__ data,  double * __restrict__  distances, int k, int dim, int centroid, int n){
  long  vector = blockIdx.x * blockDim.x +threadIdx.x;
  if (vector < n){
    double distance = 0;
    for(int dimension = 0; dimension < dim; dimension++){
      double temp = centroidCONST[dimension] - data[vector*dim+dimension];
      distance += temp*temp;
    }
    distances[(vector*k)+centroid] = sqrt(distance);
  }
}

__global__ void centroidConstantDistance2UR2(const double * __restrict__ data,  double * __restrict__  distances, int k, int dim, int centroid, int n){
  long  vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector < n){
    double distance = 0;
    for(int dimension = 0; dimension < dim; dimension+=2){
      double temp = centroidCONST[dimension] - data[vector*dim+dimension];
      distance += temp*temp;
      temp = centroidCONST[dimension+1] - data[vector*dim+dimension+1];
      distance += temp*temp;
    }
    distances[(vector*k)+centroid] = sqrt(distance);
  }
}

__global__ void centroidConstantDistance2UR4(const double * __restrict__ data,  double * __restrict__  distances, int k, int dim, int centroid, int n){
  long  vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector < n) {
    double distance = 0;
    for(int dimension = 0; dimension < dim; dimension+=4){
      double temp = centroidCONST[dimension] - data[vector*dim+dimension];
      distance += temp*temp;
      temp = centroidCONST[dimension+1] - data[vector*dim+dimension+1];
      distance += temp*temp;
      temp = centroidCONST[dimension+2] - data[vector*dim+dimension+2];
      distance += temp*temp;
      temp = centroidCONST[dimension+3] - data[vector*dim+dimension+3];
      distance += temp*temp;
    }
    distances[(vector*k)+centroid] = sqrt(distance);
  }
}

__global__ void centroidConstantDistance2UR8(const double * __restrict__ data,  double * __restrict__  distances, int k, int dim, int centroid, int n){
  long  vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector < n){
    double distance = 0;
    for(int dimension = 0; dimension < dim; dimension+=8){
      double temp = centroidCONST[dimension] - data[vector*dim+dimension];
      distance += temp*temp;
      temp = centroidCONST[dimension+1] - data[vector*dim+dimension+1];
      distance += temp*temp;
      temp = centroidCONST[dimension+2] - data[vector*dim+dimension+2];
      distance += temp*temp;
      temp = centroidCONST[dimension+3] - data[vector*dim+dimension+3];
      distance += temp*temp;
      temp = centroidCONST[dimension+4] - data[vector*dim+dimension+4];
      distance += temp*temp;
      temp = centroidCONST[dimension+5] - data[vector*dim+dimension+5];
      distance += temp*temp;
      temp = centroidCONST[dimension+6] - data[vector*dim+dimension+6];
      distance += temp*temp;
      temp = centroidCONST[dimension+7] - data[vector*dim+dimension+7];
      distance += temp*temp;
    }
    distances[(vector*k)+centroid] = sqrt(distance);
  }
}


__global__ void labelMins(int * labels, double *distances, int k, int n){
  //each thread looks at a vector, min reduction on k associated distances to centroids
  //Classify each vecotr with its closeset centroid
  //Can not imporve by adding more threads because it would require more complciated threads to calc distance
  // Too large to fit any dimesnions into constant or shared
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector < n){
    int minIndex = 0;
    double min = DBL_MAX;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labels[vector] = minIndex;
  }
}

__global__ void labelMins1(int * __restrict__ labels,const double * __restrict__ distances, int k, int n){
  //each thread looks at a vector, min reduction on k associated distances to centroids
  //Classify each vecotr with its closeset centroid
  //Can not imporve by adding more threads because it would require more complciated threads to calc distance
  // Too large to fit any dimesnions into constant or shared
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector < n){
    int minIndex = 0;
    double min = DBL_MAX;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labels[vector] = minIndex;
  }
}

__global__ void vectorLabelDistance(double * data, double *  centroids, double *  distances, int k, int dim, int n, int *labels){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension++){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
      }
      distance = sqrt(distance);
      distances[(vector*k)+centroid] = distance;
    }
  }
  int minIndex = 0;
  double min = DBL_MAX;
  
  for(int centroid = 0; centroid < k; centroid++){
    if( distances[(vector*k)+centroid] < min){
      minIndex = centroid;
      min = distances[(vector*k)+centroid];
    }
  }
  labels[vector] = minIndex;
}

__global__ void vectorLabelDistance1(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension++){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
      }
      distance = sqrt(distance);
      distances[(vector*k)+centroid] = distance;
    }
  }
  int minIndex = 0;
  double min = DBL_MAX;
  
  for(int centroid = 0; centroid < k; centroid++){
    if( distances[(vector*k)+centroid] < min){
      minIndex = centroid;
      min = distances[(vector*k)+centroid];
    }
  }
  labels[vector] = minIndex;
}

__global__ void vectorLabelDistance1UR2(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension+=2){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
	temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
	distance += temp*temp;

      }
      distance = sqrt(distance);
      distances[(vector*k)+centroid] = distance;
    }
  }
  int minIndex = 0;
  double min = DBL_MAX;
  
  for(int centroid = 0; centroid < k; centroid++){
    if( distances[(vector*k)+centroid] < min){
      minIndex = centroid;
      min = distances[(vector*k)+centroid];
    }
  }
  labels[vector] = minIndex;
}
__global__ void vectorLabelDistance1UR4(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension+=4){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
	temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+2] - data[vector*dim+dimension+2];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+3] - data[vector*dim+dimension+3];
	distance += temp*temp;
      }
      distance = sqrt(distance);
      distances[(vector*k)+centroid] = distance;
    }
  }
  int minIndex = 0;
  double min = DBL_MAX;
  
  for(int centroid = 0; centroid < k; centroid++){
    if( distances[(vector*k)+centroid] < min){
      minIndex = centroid;
      min = distances[(vector*k)+centroid];
    }
  }
  labels[vector] = minIndex;
}
__global__ void vectorLabelDistance1UR8(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension+=8){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
	temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+2] - data[vector*dim+dimension+2];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+3] - data[vector*dim+dimension+3];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+4] - data[vector*dim+dimension+4];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+5] - data[vector*dim+dimension+5];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+6] - data[vector*dim+dimension+6];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+7] - data[vector*dim+dimension+7];
	distance += temp*temp;
      }
      distance = sqrt(distance);
      distances[(vector*k)+centroid] = distance;
    }
  }
  int minIndex = 0;
  double min = DBL_MAX;
  
  for(int centroid = 0; centroid < k; centroid++){
    if( distances[(vector*k)+centroid] < min){
      minIndex = centroid;
      min = distances[(vector*k)+centroid];
    }
  }
  labels[vector] = minIndex;
}

__global__ void vectorLabelDistance1(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels, int * __restrict__ sizes){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension++){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
      }
      distance = sqrt(distance);
      distances[(vector*k)+centroid] = distance;
    }
  }
  int minIndex = 0;
  double min = DBL_MAX;
  
  for(int centroid = 0; centroid < k; centroid++){
    if( distances[(vector*k)+centroid] < min){
      minIndex = centroid;
      min = distances[(vector*k)+centroid];
    }
  }
  labels[vector] = minIndex;
  sizes[minIndex] += 1;
}

__global__ void vectorLabelDistance1UR2(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels,int * __restrict__ sizes){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension+=2){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
	temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
	distance += temp*temp;

      }
      distance = sqrt(distance);
      distances[(vector*k)+centroid] = distance;
    }
  }
  int minIndex = 0;
  double min = DBL_MAX;
  
  for(int centroid = 0; centroid < k; centroid++){
    if( distances[(vector*k)+centroid] < min){
      minIndex = centroid;
      min = distances[(vector*k)+centroid];
    }
  }
  labels[vector] = minIndex;
  sizes[minIndex] += 1;
}
__global__ void vectorLabelDistance1UR4(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels, int * __restrict__ sizes){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension+=4){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
	temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+2] - data[vector*dim+dimension+2];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+3] - data[vector*dim+dimension+3];
	distance += temp*temp;
      }
      distance = sqrt(distance);
      distances[(vector*k)+centroid] = distance;
    }
  }
  int minIndex = 0;
  double min = DBL_MAX;
  
  for(int centroid = 0; centroid < k; centroid++){
    if( distances[(vector*k)+centroid] < min){
      minIndex = centroid;
      min = distances[(vector*k)+centroid];
    }
  }
  labels[vector] = minIndex;
  sizes[minIndex] += 1;
}
__global__ void vectorLabelDistance1UR8(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels,int * __restrict__ sizes){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension+=8){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
	temp = centroids[centroid*dim+dimension+1] - data[vector*dim+dimension+1];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+2] - data[vector*dim+dimension+2];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+3] - data[vector*dim+dimension+3];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+4] - data[vector*dim+dimension+4];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+5] - data[vector*dim+dimension+5];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+6] - data[vector*dim+dimension+6];
	distance += temp*temp;
	temp = centroids[centroid*dim+dimension+7] - data[vector*dim+dimension+7];
	distance += temp*temp;
      }
      distance = sqrt(distance);
      distances[(vector*k)+centroid] = distance;
    }
  }
  int minIndex = 0;
  double min = DBL_MAX;
  
  for(int centroid = 0; centroid < k; centroid++){
    if( distances[(vector*k)+centroid] < min){
      minIndex = centroid;
      min = distances[(vector*k)+centroid];
    }
  }
  labels[vector] = minIndex;
  sizes[minIndex] += 1;
}

//Reduction sum and divide
//Atomic add and divide
//
//Reduction problem
//HORRIBLY SLOW
//AWScomputer capability is not high enough on aws
//WOuld try to implement this with thrust but no point because of compute capability
__global__ void updateMeans(double * data, double * sums, int * labels, int dim, int n){
  //sums is just like centroids/ divide all elements by n
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    //each vector looks at its
    int centroid = labels[vector];
    for(int d = 0; d < dim; d ++){
      myAtomicAdd(&(sums[centroid * dim + d]), data[vector * dim + d]);
      //atomicAdd(&sums[centroid * dim + d], 1.0);
    }
  }
}

__global__ void updateMeans1(const double * __restrict__ data, double * sums, int * labels, int dim, int n){
  //sums is just like centroids/ divide all elements by n
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    //each vector looks at its
    int centroid = labels[vector];
    for(int d = 0; d < dim; d ++){
      myAtomicAdd(&(sums[centroid * dim + d]), data[vector * dim + d]);
      //atomicAdd(&sums[centroid * dim + d], 1.0);
    }
  }
}

/*
  __global__ void assign(const float* __restrict__ data_x,
  const float* __restrict__ data_y,
  int data_size,
  const float* __restrict__ means_x,
  const float* __restrict__ means_y,
  float* __restrict__ new_sums_x,
  float* __restrict__ new_sums_y,
  int k,
  int* __restrict__ counts) {
  const int bvector = threadIdx.x;
  const int vector = blockIdx.x * blockDim.x + threadIdx.x;
  if (vector >= data_size) return;
  
  int minIndex = 0;
  double min = DBL_MAX;
  
  for(int centroid = 0; centroid < k; centroid++){
  double distance = 0;
  for(int dimension = 0; dimension < dim; dimension++){
  double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
  distance += temp * temp;
  }
  distance = sqrt(distance);
  if(distance < min){
  minIndex = centroid;
  min = distances[(vector*k)+centroid];
  }
  }
  
  __syncthreads();

  // Reduction step.

  const int x = bvector;
  const int y = bvector + blockDim.x;
  const int count = bvector + blockDim.x + blockDim.x;

  for (int centroid = 0; centroid < k; centroid++){
  // Zeros if this point (thread) is not assigned to the cluster, else the
  // values of the point.
  shared_data[x] = (best_cluster == cluster) ? x_value : 0;
  shared_data[y] = (best_cluster == cluster) ? y_value : 0;
  shared_data[count] = (best_cluster == cluster) ? 1 : 0;
  __syncthreads();

  // Tree-reduction for this cluster.
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
  if (local_index < stride) {
  for(int d = 0; d < dim; d++){
  data[x] += data[x + stride];
  }
  data[y] += data[y + stride];
  shared_data[count] += shared_data[count + stride];
  }
  __syncthreads();
  }
      
  // Now shared_data[0] holds the sum for x.

  if (bvec == 0) {
  const int centroid_index = blockIdx.x * k + centroid;
  for(int d = 0; d < dim; d++){
  new_centroids[centroid_index+d] = centroids[centroid_index+d];
  }
  sizes[cluster_index] = sizes[count];
  }
  __syncthreads();
  }
  }
  /*
  __global__ void reduce(float* __restrict__ means_x,
  float* __restrict__ means_y,
  float* __restrict__ new_sum_x,
  float* __restrict__ new_sum_y,
  int k,
  int* __restrict__ counts) {
  extern __shared__ float shared_data[];

  const int index = threadIdx.x;
  const int y_offset = blockDim.x;

  // Load into shared memory for more efficient reduction.
  shared_data[index] = new_sum_x[index];
  shared_data[y_offset + index] = new_sum_y[index];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= k; stride /= 2) {
  if (index < stride) {
  shared_data[index] += shared_data[index + stride];
  shared_data[y_offset + index] += shared_data[y_offset + index + stride];
  }
  __syncthreads();
  }

  // The first k threads can recompute their clusters' means now.
  if (index < k) {
  const int count = max(1, counts[index]);
  means_x[index] = new_sum_x[index] / count;
  means_y[index] = new_sum_y[index] / count;
  new_sum_y[index] = 0;
  new_sum_x[index] = 0;
  counts[index] = 0;
  }
  }
*/

//Dim1[sum for c1, sume for c2, sum for c3...]
//Dim2[sum for c1, sume for c2, sum for c3...]
//Dim3[sum for c1, sume for c2, sum for c3...]
//...
//All in one https://ieeexplore.ieee.org/abstract/document/6009040
//Do all in one kernel, see convergence
#endif // #ifndef KERNELS_H_
