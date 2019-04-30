#ifndef LAUNCHES_H_
#define LAUNCHES_H_

#define THREADS_PER_BLOCK 512

int * cpuGOLD(int n, int k, int iter, int dim, double * data, double * centroids){
  // Label 1, Label 2, Label 3, .....
  int *labels = (int*)malloc(n*sizeof(int));
  if (!labels) {
    fprintf(stderr, "labels unable to allocate %d ints\n", n);
    return NULL; // fail but return
  }

  // Size of centroid 1, size of centroid 2
  int *sizes = (int*)malloc(k*sizeof(int));
  if (!sizes) {
    fprintf(stderr, "sizes unable to allocate %d ints\n", dim);
    return NULL; // fail but return
  }
 
  //Distance N=0/Centroid = 0, Distance N=0/Centroid = 1, ..., Distance N=0/Centroid = k, Distance N=1/Centroid = 0, .....
  double* distances = (double*)malloc(n*k*sizeof(double));
  if (!distances) {
    fprintf(stderr, "distances  unable to allocate %d x %d doubles\n", n, k);
    return NULL; // fail but return
  }

  /*TWO STEP PROCESS
   *Classify
   *Find Means */
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    for(int vector = 0; vector < n; vector++){
      for(int centroid = 0; centroid < k; centroid++){
	double distance = 0;
	for(int dimension = 0; dimension < dim; dimension++){
	  double temp = centroids[centroid*dim+dimension] - data[vector*dim+dimension];
	  distance += temp*temp;
	}
	distance = sqrt(distance);
	distances[(vector*k)+centroid] = distance;
      }
    }
    
    //Reset sizes
    for(int i = 0;i < k; i ++){
      sizes[i] = 1;
    }
    
    //Classify each vector
    for(int vector = 0; vector < n; vector++){
      int minIndex = 0;
      double min = 9999999;
      
      for(int centroid = 0; centroid < k; centroid++){
	if( distances[(vector*k)+centroid] < min){
	  minIndex = centroid;
	  min = distances[(vector*k)+centroid];
	}
      }
      labels[vector] = minIndex;
    }
    
    //Go through each vector and update the mean of the centroid it is classified as
    for(int vector = 0; vector < n; vector++){
      sizes[labels[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroids[labels[vector]*dim+d] = (centroids[labels[vector]*dim+d] * (sizes[labels[vector]]-1) + data[vector*dim+d]) / sizes[labels[vector]];
      }
    }
    //Iterate
    b++;
  }
  //{v1,v1,v2,v2,v3,v3}
  // 0  1  2  3  4  5
  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labels[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }
  free(sizes);
  free(distances);
  return labels;
}

int * launchVectorDistance(int * labelsGPU, double * data, double *distances,double * centroidsGPU, int *sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));
  
  double *distancesD, *centroidsD, *dataD;
  cudaError_t code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&centroidsD, k * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    cudaMemcpy(centroidsD, centroidsGPU, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);
    
    vectorDistance<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n);

    cudaMemcpy(distances, distancesD, n * k * sizeof(double), cudaMemcpyDeviceToHost);

 
    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }

    //Classify each vecotr with its closeset centroid
    for(int vector = 0; vector < n; vector++){
      int minIndex = 0;
      double min = 9999999;
      
      for(int centroid = 0; centroid < k; centroid++){
	if( distances[(vector*k)+centroid] < min){
	  minIndex = centroid;
	  min = distances[(vector*k)+centroid];
	}
      }
      labelsGPU[vector] = minIndex;
    }

    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }
  
  cudaFree(distancesD);
  cudaFree(centroidsD);
  cudaFree(dataD);
  
  return labelsGPU;
}

int * launchVectorDistanceUR(int * labelsGPU, double * data, double *distances,double * centroidsGPU, int * sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));
  
  double *distancesD, *centroidsD, *dataD;
  cudaError_t code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&centroidsD, k * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    cudaMemcpy(centroidsD, centroidsGPU, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);

    if(dim%8==0){
      vectorDistance2UR8<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n);
    }else if(dim%4==0){
      vectorDistance2UR4<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n);
    }else if(dim%2==0){
      vectorDistance2UR2<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n);
    }else{
      vectorDistance2<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n);
    }
    cudaMemcpy(distances, distancesD, n * k * sizeof(double), cudaMemcpyDeviceToHost);

 
    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }

    //Classify each vecotr with its closeset centroid
    for(int vector = 0; vector < n; vector++){
      int minIndex = 0;
      double min = 9999999;
      
      for(int centroid = 0; centroid < k; centroid++){
	if( distances[(vector*k)+centroid] < min){
	  minIndex = centroid;
	  min = distances[(vector*k)+centroid];
	}
      }
      labelsGPU[vector] = minIndex;
    }

    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }
  
  cudaFree(distancesD);
  cudaFree(centroidsD);
  cudaFree(dataD);
  
  return labelsGPU;
}

int * launchCentroidDistanceUR(int * labelsGPU, double * data, double *distances, double* centroidsGPU, int *sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));

  double *distancesD, *centroidsD, *dataD;
  
  cudaError_t code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&centroidsD, k * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    cudaMemcpy(centroidsD, centroidsGPU, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);
    //DO i actually need to copy over distances?
    for(int centroid = 0; centroid < k; centroid ++){
      if(dim%8==0){
	centroidDistance2UR8<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, centroid, n);
      }else if(dim%4==0){	
	centroidDistance2UR4<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, centroid, n);
      }else if(dim%2==0){
	centroidDistance2UR2<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, centroid, n);
      }else{
	centroidDistance2<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, centroid, n);
      }
    }
    cudaMemcpy(distances, distancesD, n * k * sizeof(double), cudaMemcpyDeviceToHost);

    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }

    //Classify each vecotr with its closeset centroid
    for(int vector = 0; vector < n; vector++){
      int minIndex = 0;
      double min = 9999999;
      
      for(int centroid = 0; centroid < k; centroid++){
	if( distances[(vector*k)+centroid] < min){
	  minIndex = centroid;
	  min = distances[(vector*k)+centroid];
	}
      }
      labelsGPU[vector] = minIndex;
    }

    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }

  cudaFree(distancesD);
  cudaFree(centroidsD);
  cudaFree(dataD);
  return labelsGPU;
}

int * launchCentroidDistance(int * labelsGPU, double * data, double *distances, double* centroidsGPU, int * sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));

  double *distancesD, *centroidsD, *dataD;
  
  cudaError_t code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&centroidsD, k * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    cudaMemcpy(centroidsD, centroidsGPU, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);
    //DO i actually need to copy over distances?
    for(int centroid = 0; centroid < k; centroid ++){
      centroidDistance<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, centroid, n);
    }
    cudaMemcpy(distances, distancesD, n * k * sizeof(double), cudaMemcpyDeviceToHost);

    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }

    //Classify each vecotr with its closeset centroid
    for(int vector = 0; vector < n; vector++){
      int minIndex = 0;
      double min = 9999999;
      
      for(int centroid = 0; centroid < k; centroid++){
	if( distances[(vector*k)+centroid] < min){
	  minIndex = centroid;
	  min = distances[(vector*k)+centroid];
	}
      }
      labelsGPU[vector] = minIndex;
    }

    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }

  cudaFree(distancesD);
  cudaFree(centroidsD);
  cudaFree(dataD);
  return labelsGPU;
}

int * launchCentroidConstantDistanceUR(int * labelsGPU, double * data, double *distances, double* centroidsGPU, int * sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));

  double *distancesD, *dataD;
  
  cudaError_t code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    //cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);
    //DO I actually need to copy over distances?
    for(int centroid = 0; centroid < k; centroid ++){
      cudaMemcpyToSymbol(centroidCONST,  centroidsGPU+centroid*dim, sizeof(double)*dim);
      if(dim%8==0){
	centroidConstantDistance2UR8<<<dimGrid, dimBlock>>>(dataD, distancesD, k, dim, centroid, n);
      }else if(dim%4==0){	
	centroidConstantDistance2UR4<<<dimGrid, dimBlock>>>(dataD, distancesD, k, dim, centroid, n);
      }else if(dim%2==0){
	centroidConstantDistance2UR2<<<dimGrid, dimBlock>>>(dataD, distancesD, k, dim, centroid, n);
      }else{
	centroidConstantDistance2<<<dimGrid, dimBlock>>>(dataD, distancesD, k, dim, centroid, n);
      }

    }
    cudaMemcpy(distances, distancesD, n * k * sizeof(double), cudaMemcpyDeviceToHost);
    
    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }

    //Classify each vecotr with its closeset centroid
    for(int vector = 0; vector < n; vector++){
      int minIndex = 0;
      double min = 9999999;
      
      for(int centroid = 0; centroid < k; centroid++){
	if( distances[(vector*k)+centroid] < min){
	  minIndex = centroid;
	  min = distances[(vector*k)+centroid];
	}
      }
      labelsGPU[vector] = minIndex;
    }

    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }
  cudaFree(distancesD);
  cudaFree(dataD);
  return labelsGPU;
}

int * launchCentroidConstantDistance(int * labelsGPU, double * data, double *distances, double* centroidsGPU, int * sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));

  double *distancesD, *dataD;
  
  cudaError_t code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    //cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);
    //DO I actually need to copy over distances?
    for(int centroid = 0; centroid < k; centroid ++){
      cudaMemcpyToSymbol(centroidCONST,  centroidsGPU+centroid*dim, sizeof(double)*dim);

      centroidConstantDistance<<<dimGrid, dimBlock>>>(dataD, distancesD, k, dim, centroid, n);
    }
    cudaMemcpy(distances, distancesD, n * k * sizeof(double), cudaMemcpyDeviceToHost);
    
    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }

    //Classify each vecotr with its closeset centroid
    for(int vector = 0; vector < n; vector++){
      int minIndex = 0;
      double min = 9999999;
      
      for(int centroid = 0; centroid < k; centroid++){
	if( distances[(vector*k)+centroid] < min){
	  minIndex = centroid;
	  min = distances[(vector*k)+centroid];
	}
      }
      labelsGPU[vector] = minIndex;
    }

    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }
  cudaFree(distancesD);
  cudaFree(dataD);
  return labelsGPU;
}

int * launchLabelMins(int * labelsGPU, double * data, double *distances, double* centroidsGPU, int *sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));
  
  int* labelsD;
  double *distancesD;
  cudaError_t code = cudaMalloc(&labelsD, n * sizeof(int));
  if(code!=cudaSuccess){
    fprintf(stderr,"LabelsD GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"DistancesD GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    
    for(int vector = 0; vector < n; vector++){
      for(int centroid = 0; centroid < k; centroid++){
	double distance = 0;
	for(int dimension = 0; dimension < dim; dimension++){
	  double temp = centroidsGPU[centroid*dim+dimension] - data[vector*dim+dimension];
	  distance += temp*temp;
	}
	distance = sqrt(distance);
	distances[(vector*k)+centroid] = distance;
      }
    }
 
    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }
    
    cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);
    labelMins1<<<dimGrid, dimBlock>>>(labelsD, distancesD, k, n);
    cudaMemcpy(labelsGPU, labelsD, n * sizeof(int), cudaMemcpyDeviceToHost);
    //Classify each vecotr with its closeset centroid
    for(int vector = 0; vector < n; vector++){
      int minIndex = 0;
      double min = 9999999;
      
      for(int centroid = 0; centroid < k; centroid++){
	if( distances[(vector*k)+centroid] < min){
	  minIndex = centroid;
	  min = distances[(vector*k)+centroid];
	}
      }
      labelsGPU[vector] = minIndex;
    }
    
    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //iterate
    b++;
  }//END WHILE
  
  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }
  
  cudaFree(distancesD);
  cudaFree(labelsD);
  
  return labelsGPU;
}

int * launchThreadPerVectorWithLabels(int * labelsGPU, double * data, double *distances,double* centroidsGPU, int * sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));

  int* labelsD;
  double *distancesD, *centroidsD, *dataD;
  cudaError_t   code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&centroidsD, k * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&labelsD, n * sizeof(int));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    cudaMemcpy(centroidsD, centroidsGPU, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(centroidsD, centroidsGPU, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);
    
    vectorLabelDistance<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n, labelsD);

    cudaMemcpy(labelsGPU, labelsD, n * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(distances, distancesD, n * k * sizeof(double), cudaMemcpyDeviceToHost);

    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }

    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }

  cudaFree(distancesD);
  cudaFree(centroidsD);
  cudaFree(dataD);
  cudaFree(labelsD);

  return labelsGPU;
}

int * launchThreadPerVectorWithLabelsUR(int * labelsGPU, double * data, double *distances,double* centroidsGPU, int * sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));

  int* labelsD;
  double *distancesD, *centroidsD, *dataD;
  cudaError_t   code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&centroidsD, k * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&labelsD, n * sizeof(int));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    cudaMemcpy(centroidsD, centroidsGPU, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(centroidsD, centroidsGPU, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);

    if(dim%8==0){
       vectorLabelDistance1UR8<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n, labelsD);
    }else if(dim%4==0){
      vectorLabelDistance1UR4<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n, labelsD);
    }else if(dim%2==0){
      vectorLabelDistance1UR2<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n, labelsD);
    }else{
      vectorLabelDistance1<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n, labelsD);
    }
    cudaMemcpy(labelsGPU, labelsD, n * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(distances, distancesD, n * k * sizeof(double), cudaMemcpyDeviceToHost);

    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }

    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }

  cudaFree(distancesD);
  cudaFree(centroidsD);
  cudaFree(dataD);
  cudaFree(labelsD);

  return labelsGPU;
}

int * launchCentroidConstantLabels(int * labelsGPU, double * data, double *distances,double* centroidsGPU, int * sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));

  int* labelsD;
  double *distancesD, *centroidsD, *dataD;
  
  cudaError_t   code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&centroidsD, k * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMalloc(&labelsD, n * sizeof(int));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    for(int centroid = 0; centroid < k; centroid ++){
      cudaMemcpyToSymbol(centroidCONST,  (centroidsGPU+centroid*dim), sizeof(double)*dim);
      centroidConstantDistance<<<dimGrid, dimBlock>>>(dataD, distancesD, k, dim, centroid, n);
    }
    cudaMemcpy(distances, distancesD, n * k * sizeof(double), cudaMemcpyDeviceToHost);

    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }

    cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);
    labelMins<<<dimGrid, dimBlock>>>(labelsD, distancesD, k, n);
    cudaMemcpy(labelsGPU, labelsD, n * sizeof(int), cudaMemcpyDeviceToHost);

    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //for(int d = 0; d < dim; d ++){
    //  fprintf(stderr, "centroidsGPU[%d]=%f\n", d, centroidsGPU[d]);
    // }
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }
 

  cudaFree(distancesD);
  cudaFree(centroidsD);
  cudaFree(dataD);
  cudaFree(labelsD);

  return labelsGPU;
}

int * launchCentroidConstantLabelsUR(int * labelsGPU, double * data, double *distances,double* centroidsGPU, int * sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));

  int* labelsD;
  double *distancesD, *centroidsD, *dataD;
  
  cudaError_t   code = cudaMalloc(&distancesD, n * k * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&centroidsD, k * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMalloc(&labelsD, n * sizeof(int));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    for(int centroid = 0; centroid < k; centroid ++){
      cudaMemcpyToSymbol(centroidCONST,  (centroidsGPU+centroid*dim), sizeof(double)*dim);
      if(dim%8==0){
	centroidConstantDistance2UR8<<<dimGrid, dimBlock>>>(dataD, distancesD, k, dim, centroid, n);
      }else if(dim%4==0){	
	centroidConstantDistance2UR4<<<dimGrid, dimBlock>>>(dataD, distancesD, k, dim, centroid, n);
      }else if(dim%2==0){
	centroidConstantDistance2UR2<<<dimGrid, dimBlock>>>(dataD, distancesD, k, dim, centroid, n);
      }else{
	centroidConstantDistance2<<<dimGrid, dimBlock>>>(dataD, distancesD, k, dim, centroid, n);
      }
    }
    cudaMemcpy(distances, distancesD, n * k * sizeof(double), cudaMemcpyDeviceToHost);

    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 1;
    }

    cudaMemcpy(distancesD, distances, n * k * sizeof(double), cudaMemcpyHostToDevice);
    labelMins1<<<dimGrid, dimBlock>>>(labelsD, distancesD, k, n);
    cudaMemcpy(labelsGPU, labelsD, n * sizeof(int), cudaMemcpyDeviceToHost);

    //Go through each vector and update the mean of the classified centroid
    for(int vector = 0; vector < n; vector++){
      sizes[labelsGPU[vector]] += 1;
      for(int d = 0; d < dim; d++){
	centroidsGPU[labelsGPU[vector]*dim+d] = (centroidsGPU[labelsGPU[vector]*dim+d] * (sizes[labelsGPU[vector]]-1) + data[vector*dim+d]) / sizes[labelsGPU[vector]];
      }
    }
    //for(int d = 0; d < dim; d ++){
    //  fprintf(stderr, "centroidsGPU[%d]=%f\n", d, centroidsGPU[d]);
    // }
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }
 

  cudaFree(distancesD);
  cudaFree(centroidsD);
  cudaFree(dataD);
  cudaFree(labelsD);

  return labelsGPU;
}

int * launchUpdateMeans(int * labelsGPU, double * data, double *distances,double* centroidsGPU, int * sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));

  int *labelsD;
  double *centroidsD, *dataD;
  
  cudaError_t   code = cudaMalloc(&centroidsD, k * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMalloc(&labelsD, n * sizeof(int));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  
  int b = 0;
  while(b < iter){
    for(int vector = 0; vector < n; vector++){
      for(int centroid = 0; centroid < k; centroid++){
	double distance = 0;
	for(int dimension = 0; dimension < dim; dimension++){
	  double temp = centroidsGPU[centroid*dim+dimension] - data[vector*dim+dimension];
	  distance += temp*temp;
	}
	distance = sqrt(distance);
	distances[(vector*k)+centroid] = distance;
      }
    }
    //Reset sizes
    for(int i = 0; i < k; i ++){
      sizes[i] = 0;
    }

    //Classify each vector with its closeset centroid
    for(int vector = 0; vector < n; vector++){
      int minIndex = 0;
      double min = 9999999;
      
      for(int centroid = 0; centroid < k; centroid++){
	if( distances[(vector*k)+centroid] < min){
	  minIndex = centroid;
	  min = distances[(vector*k)+centroid];
	}
      }
      labelsGPU[vector] = minIndex;
      sizes[minIndex] += 1;
    }

    //Load Labels
    cudaMemcpy(labelsD, labelsGPU, n * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(centroidsD, centroidsGPU, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(centroidsD, 0, k * dim * sizeof(double));
    updateMeans1<<<dimGrid, dimBlock>>>(dataD, centroidsD, labelsD, dim, n);
    cudaMemcpy(centroidsGPU, centroidsD, k * dim * sizeof(double), cudaMemcpyDeviceToHost);
    //Load back sums
    
    for(int i = 0; i < k; i ++){
      double size = (double)sizes[i];
      for(int d = 0; d < dim; d ++){
	centroidsGPU[i*dim+d] = centroidsGPU[i*dim+d]/size;
      }
    }
    //for(int d = 0; d < dim; d ++){
      //  fprintf(stderr, "centroidsGPU[%d]=%f\n", d, centroidsGPU[d]);
    //}
    //iterate
    b++;
  }//END WHILE

  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }

  cudaFree(centroidsD);
  cudaFree(dataD);

  return labelsGPU;
}

int * launchGPU(int * labelsGPU, double * data, double *distances,double* centroidsGPU, int * sizes, int n, int k, int iter, int dim){
  dim3 dimBlock(THREADS_PER_BLOCK);
  dim3 dimGrid(ceil((float)n/(float)THREADS_PER_BLOCK));

  int *labelsD;
  int *sizesD;
  double  *distancesD, *centroidsD, *dataD;
  
  cudaError_t   code = cudaMalloc(&centroidsD, k * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  code = cudaMalloc(&dataD, n * dim * sizeof(double));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMalloc(&labelsD, n * sizeof(int));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMalloc(&sizesD, k * sizeof(int));
  if(code!=cudaSuccess){
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorName(code));
  }
  cudaMemcpy(dataD,data, n * dim * sizeof(double), cudaMemcpyHostToDevice);
  
  int b = 0;
  while(b < iter){
    //for each vector
    //find the distance to each centroid
    cudaMemcpy(centroidsD, centroidsGPU, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(sizesD, 0, k * sizeof(int));
    if(dim%8==0){
      vectorLabelDistance1UR8<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n, labelsD, sizesD);
    }else if(dim%4==0){
      vectorLabelDistance1UR4<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n, labelsD, sizesD);
    }else if(dim%2==0){
      vectorLabelDistance1UR2<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n, labelsD, sizesD);
    }else{
      vectorLabelDistance1<<<dimGrid, dimBlock>>>(dataD, centroidsD, distancesD, k, dim, n, labelsD, sizesD);
    }
    
    cudaMemset(centroidsD, 0, k * dim * sizeof(double));
    updateMeans1<<<dimGrid, dimBlock>>>(dataD, centroidsD, labelsD, dim, n);
    cudaMemcpy(centroidsGPU, centroidsD, k * dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sizes, sizesD, k * sizeof(int), cudaMemcpyDeviceToHost);
    //Load back sums
    
    for(int i = 0; i < k; i ++){
      double size = (double)sizes[i];
      for(int d = 0; d < dim; d ++){
	centroidsGPU[i*dim+d] = centroidsGPU[i*dim+d]/size;
      }
    }
    b++;
  }//END WHILE
  
  for(int vector = 0; vector < n; vector++){
    int minIndex = 0;
    double min = 9999999;
    
    for(int centroid = 0; centroid < k; centroid++){
      if( distances[(vector*k)+centroid] < min){
	minIndex = centroid;
	min = distances[(vector*k)+centroid];
      }
    }
    labelsGPU[vector] = minIndex;
    //printf("%d,%d,%d\n",(int)data[vector*dim],(int)data[vector*dim+1], minIndex);
    //printf("%d,%d\n",vector, minIndex); 
  }

  cudaFree(centroidsD);
  cudaFree(dataD);
  
  return labelsGPU;
}
//IMPLEMENT
int * launchGPUReduction(){
  return NULL;
}

#endif // #ifndef LAUNCHES_H_
