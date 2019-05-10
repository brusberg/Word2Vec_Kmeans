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

__global__ void centroidConstantDistanceNSQR(double * data,  double *  distances, int k, int dim, int centroid, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector < n){
    double distance = 0;
    for(int dimension = 0; dimension < dim; dimension++){
      double temp = centroidCONST[dimension] - data[vector*dim+dimension];
      distance += temp*temp;
    }
    distances[(vector*k)+centroid] = distance;
  }
}

//Not as fast, also weird cast rounding depending on the case, extra casts makes it slower
__global__ void centroidConstantDistanceNSQ(double * data,  double *  distances, int k, int dim, int centroid, int n){
  long vector = blockIdx.x * blockDim.x +threadIdx.x;
  if(vector < n){
    double distance = 0;
    for(int dimension = 0; dimension < dim; dimension++){
      double temp = centroidCONST[dimension] - data[vector*dim+dimension];
      temp = (*(long long *)&temp) & 0x7fffffffffffffff;
      distance +=  (double)temp;
    }
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

__global__ void vectorLabelDistance2(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels, int * __restrict__ sizes){
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    for(int centroid = 0; centroid < k; centroid++){
      double distance = 0;
      for(int dimension = 0; dimension < dim; dimension++){
	double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	distance += temp * temp;
      }
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

__global__ void vectorLabelDistance2UR2(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels,int * __restrict__ sizes){
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
__global__ void vectorLabelDistance2UR4(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels, int * __restrict__ sizes){
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
__global__ void vectorLabelDistance2UR8(const double * __restrict__ data, double * __restrict__  centroids, double * __restrict__ distances, int k, int dim, int n, int * __restrict__ labels, int * __restrict__ sizes){
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
 
__global__ void reduceToBuffer(const double * __restrict__ data, double * buffer, int * labels, int dim, int n, int k){
  //sums is just like centroids/ divide all elements by n
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    if(labels[vector] == k){
      for(int d = 0; d < dim; d ++){
	buffer[vector*dim+d] = data[vector * dim + d];
      }       
    }else{
      for(int d = 0; d < dim; d ++){
	buffer[vector*dim+d] = 0;
      }
    }
  }
}

__global__ void reduce(double * buffer, double * __restrict__  centroids, int * labels, int dim, int n, int k, int level, int * __restrict__ sizes){
  //sums is just like centroids/ divide all elements by n
  //Stores the actual threadId
  unsigned int vector = (blockIdx.x * blockDim.x) + threadIdx.x;
  //Stores the index in the array the thread is accessing
  unsigned int index = vector * level;
  //Stores the maximum index of a thread in the same block
  unsigned int end = ((blockIdx.x * blockDim.x) + blockDim.x - 1) * level;
  //Replaces the maximum index with the end of the array for the last block
  if (end > n){
    end = n;
  }
  //Keep cutting the array in half and adding the halfs together
  for(unsigned int stride = blockDim.x >> 1; stride > 0; stride >>=1){
    //Makes sure all additions are completed before going on to the next iteration
    __syncthreads();
    //Calculates the stride with respect to the level (spacing between adjacent elements)
    unsigned int real_stride = stride * level;
    //Cjecks if the thread is in the first half of the remaining array and that the corresponding element exists in the second half
    if (threadIdx.x < stride && index + real_stride <= end){
      //Adds the "i"th element of the first half with the "i"th element of the second half
      for(int d = 0; d < dim; d ++){
	buffer[index*dim+d] = buffer[(index+real_stride) * dim + d];
      }
    }
  }
    
  if(vector == 0){
    int size = sizes[k];
    for(int d = 0; d < dim; d ++){
      centroids[k*dim+d] = buffer[d]/size;
    }       
  }
}

__global__ void reduceToBufferUR2(const double * __restrict__ data, double * buffer, int * labels, int dim, int n, int k){
  //sums is just like centroids/ divide all elements by n
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    if(labels[vector] == k){ 
      for(int d = 0; d < dim; d += 2){
	buffer[vector*dim+d] = data[vector * dim + d];
	buffer[vector*dim+d+1] = data[vector * dim + d+1];
      }       
    }else{
      for(int d = 0; d < dim; d += 2){
	buffer[vector*dim+d] = 0;
	buffer[vector*dim+d+1] = 0;
      }
    }
  }
}

__global__ void reduceUR2(double * buffer, double * __restrict__  centroids, int * labels, int dim, int n, int k, int level, int * __restrict__ sizes){
  //sums is just like centroids/ divide all elements by n
  //Stores the actual threadId
  unsigned int vector = (blockIdx.x * blockDim.x) + threadIdx.x;
  //Stores the index in the array the thread is accessing
  unsigned int index = vector * level;
  //Stores the maximum index of a thread in the same block
  unsigned int end = ((blockIdx.x * blockDim.x) + blockDim.x - 1) * level;
  //Replaces the maximum index with the end of the array for the last block
  if (end > n){
    end = n;
  }
  //Keep cutting the array in half and adding the halfs together
  for(unsigned int stride = blockDim.x >> 1; stride > 0; stride >>=1){
    //Makes sure all additions are completed before going on to the next iteration
    __syncthreads();
    //Calculates the stride with respect to the level (spacing between adjacent elements)
    unsigned int real_stride = stride * level;
    //Cjecks if the thread is in the first half of the remaining array and that the corresponding element exists in the second half
    if (threadIdx.x < stride && index + real_stride <= end){
      //Adds the "i"th element of the first half with the "i"th element of the second half
      for(int d = 0; d < dim; d += 2){
	buffer[index*dim+d] = buffer[(index+real_stride) * dim + d];
	buffer[index*dim+d+1] = buffer[(index+real_stride+1) * dim + d];
      }
    }
  }
    
  if(vector == 0){
    int size = sizes[k];
    for(int d = 0; d < dim; d += 2){
      centroids[k*dim+d] = buffer[d]/size;
      centroids[k*dim+d+1] = buffer[d+1]/size;
    }       
  }
}

__global__ void reduceToBufferUR4(const double * __restrict__ data, double * buffer, int * labels, int dim, int n, int k){
  //sums is just like centroids/ divide all elements by n
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    if(labels[vector] == k){
      for(int d = 0; d < dim; d += 4){
	buffer[vector*dim+d] = data[vector * dim + d];
	buffer[vector*dim+d+1] = data[vector * dim + d+1];
	buffer[vector*dim+d+2] = data[vector * dim + d + 2];
	buffer[vector*dim+d+3] = data[vector * dim + d+3];
      }       
    }else{
      for(int d = 0; d < dim; d += 4){
	buffer[vector*dim+d] = 0;
	buffer[vector*dim+d+1] = 0;
	buffer[vector*dim+d+2] = 0;
	buffer[vector*dim+d+3] = 0;
      }
    }
  }
}

__global__ void reduceUR4(double * buffer, double * __restrict__  centroids, int * labels, int dim, int n, int k, int level, int * __restrict__ sizes){
  //sums is just like centroids/ divide all elements by n
  //Stores the actual threadId
  unsigned int vector = (blockIdx.x * blockDim.x) + threadIdx.x;
  //Stores the index in the array the thread is accessing
  unsigned int index = vector * level;
  //Stores the maximum index of a thread in the same block
  unsigned int end = ((blockIdx.x * blockDim.x) + blockDim.x - 1) * level;
  //Replaces the maximum index with the end of the array for the last block
  if (end > n){
    end = n;
  }
  //Keep cutting the array in half and adding the halfs together
  for(unsigned int stride = blockDim.x >> 1; stride > 0; stride >>=1){
    //Makes sure all additions are completed before going on to the next iteration
    __syncthreads();
    //Calculates the stride with respect to the level (spacing between adjacent elements)
    unsigned int real_stride = stride * level;
    //Cjecks if the thread is in the first half of the remaining array and that the corresponding element exists in the second half
    if (threadIdx.x < stride && index + real_stride <= end){
      //Adds the "i"th element of the first half with the "i"th element of the second half
      for(int d = 0; d < dim; d += 4){
	buffer[index*dim+d] = buffer[(index+real_stride) * dim + d];
	buffer[index*dim+d+1] = buffer[(index+real_stride+1) * dim + d];
	buffer[index*dim+d+2] = buffer[(index+real_stride+2) * dim + d];
	buffer[index*dim+d+3] = buffer[(index+real_stride+3) * dim + d];
      }
    }
  }
    
  if(vector == 0){
    int size = sizes[k];
    for(int d = 0; d < dim; d += 4){
      centroids[k*dim+d] = buffer[d]/size;
      centroids[k*dim+d+1] = buffer[d+1]/size;
      centroids[k*dim+d+2] = buffer[d+2]/size;
      centroids[k*dim+d+3] = buffer[d+3]/size;
    }       
  }
}

__global__ void reduceToBufferUR8(const double * __restrict__ data, double * buffer, int * labels, int dim, int n, int k){
  //sums is just like centroids/ divide all elements by n
  long vector = blockIdx.x * blockDim.x + threadIdx.x;
  if(vector < n){
    if(labels[vector] == k){
      for(int d = 0; d < dim; d += 8){
	buffer[vector*dim+d] = data[vector * dim + d];
	buffer[vector*dim+d+1] = data[vector * dim + d+1];
	buffer[vector*dim+d+2] = data[vector * dim + d+2];
	buffer[vector*dim+d+3] = data[vector * dim + d+3];
	buffer[vector*dim+d+4] = data[vector * dim + d+4];
	buffer[vector*dim+d+5] = data[vector * dim + d+5];
	buffer[vector*dim+d+6] = data[vector * dim + d+6];
	buffer[vector*dim+d+7] = data[vector * dim + d+7];
      }       
    }else{
      for(int d = 0; d < dim; d += 8){
	buffer[vector*dim+d] = 0;
	buffer[vector*dim+d+1] = 0;
	buffer[vector*dim+d+2] = 0;
	buffer[vector*dim+d+3] = 0;
	buffer[vector*dim+d+4] = 0;
	buffer[vector*dim+d+5] = 0;
	buffer[vector*dim+d+6] = 0;
	buffer[vector*dim+d+7] = 0;
      }
    }
  }
}

__global__ void reduceUR8(double * buffer, double * __restrict__  centroids, int * labels, int dim, int n, int k, int level, int * __restrict__ sizes){
  //sums is just like centroids/ divide all elements by n
  //Stores the actual threadId
  unsigned int vector = (blockIdx.x * blockDim.x) + threadIdx.x;
  //Stores the index in the array the thread is accessing
  unsigned int index = vector * level;
  //Stores the maximum index of a thread in the same block
  unsigned int end = ((blockIdx.x * blockDim.x) + blockDim.x - 1) * level;
  //Replaces the maximum index with the end of the array for the last block
  if (end > n){
    end = n;
  }
  //Keep cutting the array in half and adding the halfs together
  for(unsigned int stride = blockDim.x >> 1; stride > 0; stride >>=1){
    //Makes sure all additions are completed before going on to the next iteration
    __syncthreads();
    //Calculates the stride with respect to the level (spacing between adjacent elements)
    unsigned int real_stride = stride * level;
    //Cjecks if the thread is in the first half of the remaining array and that the corresponding element exists in the second half
    if (threadIdx.x < stride && index + real_stride <= end){
      //Adds the "i"th element of the first half with the "i"th element of the second half
      for(int d = 0; d < dim; d += 4){
	buffer[index*dim+d] = buffer[(index+real_stride) * dim + d];
	buffer[index*dim+d+1] = buffer[(index+real_stride+1) * dim + d];
	buffer[index*dim+d+2] = buffer[(index+real_stride+2) * dim + d];
	buffer[index*dim+d+3] = buffer[(index+real_stride+3) * dim + d];
	buffer[index*dim+d+4] = buffer[(index+real_stride+4) * dim + d];
	buffer[index*dim+d+5] = buffer[(index+real_stride+5) * dim + d];
	buffer[index*dim+d+6] = buffer[(index+real_stride+6) * dim + d];
	buffer[index*dim+d+7] = buffer[(index+real_stride+7) * dim + d];
      }
    }
  }
    
  if(vector == 0){
    int size = sizes[k];
    for(int d = 0; d < dim; d += 8){
      centroids[k*dim+d] = buffer[d]/size;
      centroids[k*dim+d+1] = buffer[d+1]/size;
      centroids[k*dim+d+2] = buffer[d+2]/size;
      centroids[k*dim+d+3] = buffer[d+3]/size;
      centroids[k*dim+d+4] = buffer[d+4]/size;
      centroids[k*dim+d+5] = buffer[d+5]/size;
      centroids[k*dim+d+6] = buffer[d+6]/size;
      centroids[k*dim+d+7] = buffer[d+7]/size;
    }       
  }
}
#endif // #ifndef KERNELS_H_
