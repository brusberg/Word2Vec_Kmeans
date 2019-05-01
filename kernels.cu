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

__global__ void labelMins1(int * __restrict__ labels,const double * __restrict__ distances, int k, int n, int * sizes){
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
    sizes[minIndex] += 1;
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
}

//Reduction sum and divide
//Atomic add and divide
//
//Reduction problem
//HORRIBLY SLOW
//AWScomputer capability is not high enough on aws
//Wpuld try to implement this with thrust but no point because of compute capability
__global__ void updateMeans(double * data, double * sums, int * labels, int dim, int n){
  //sums is just like centroids/ divide all elements by n
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    //each vector looks at its
    int centroid = labels[vector];
    for(int d = 0; d < dim; d ++){
      myAtomicAdd(&(sums[centroid * dim + d]), data[vector * dim + d]);
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

__global__ void fine_reduce(const float* __restrict__ data,
                            int n,
                            const float* __restrict__ means, 
                            float* __restrict__ new_sums,
                            int k,
                            int* __restrict__ counts) {
  extern __shared__ float shared_data[];

  const int local_index = threadIdx.x;
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index >= n) return;

  // Load the mean values into shared memory.
  if (local_index < k) {
    buffer[local_index] = means[local_index];
    buffer[k + local_index] = means_y[local_index];
  }

  __syncthreads();

  // Load once here.
  const float x_value = data[global_index];
  const float y_value = data_y[global_index];

  float best_distance = FLT_MAX;
  int best_cluster = -1;
  for (int cluster = 0; cluster < k; ++cluster) {
    const float distance = squared_l2_distance(x_value,
                                               y_value,
                                               buffer[cluster],
                                               buffer[k + cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  __syncthreads();

  // reduction

  const int x = local_index;
  const int y = local_index + blockDim.x;
  const int count = local_index + blockDim.x + blockDim.x;

  for (int cluster = 0; cluster < k; ++cluster) {
    buffer[x] = (best_cluster == cluster) ? x_value : 0;
    buffer[y] = (best_cluster == cluster) ? y_value : 0;
    buffer[count] = (best_cluster == cluster) ? 1 : 0;
    __syncthreads();

    // Reduction for this cluster.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (local_index < stride) {
        buffer[x] += buffer[x + stride];
        buffer[y] += buffer[y + stride];
        buffer[count] += buffer[count + stride];
      }
      __syncthreads();
    }

    // Now buffer[0] holds the sum for x.

    if (local_index == 0) {
      const int cluster_index = blockIdx.x * k + cluster;
      new_sums_x[cluster_index] = buffer[x];
      new_sums_y[cluster_index] = buffer[y];
      counts[cluster_index] = buffer[count];
    }
    __syncthreads();
  }
}

__global__ void coarse_reduce(float* __restrict__ means,
                              float* __restrict__ means_y,
                              float* __restrict__ new_sum_x,
                              float* __restrict__ new_sum_y,
                              int k,
                              int* __restrict__ counts) {
  extern __shared__ float buffer[];

  const int index = threadIdx.x;
  const int y_offset = blockDim.x;

  buffer[index] = new_sum_x[index];
  buffer[y_offset + index] = new_sum_y[index];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= k; stride /= 2) {
    if (index < stride) {
      buffer[index] += buffer[index + stride];
      buffer[y_offset + index] += buffer[y_offset + index + stride];
    }
    __syncthreads();
  }

//Dim1[sum for c1, sume for c2, sum for c3...]
//Dim2[sum for c1, sume for c2, sum for c3...]
//Dim3[sum for c1, sume for c2, sum for c3...]
//...
//All in one https://ieeexplore.ieee.org/abstract/document/6009040
//Do all in one kernel, see convergence
#endif // #ifndef KERNELS_H_
