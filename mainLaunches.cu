/* Brenden Brusberg
 * Main implementation for k-means on higher dimensional data
 * CS - 677, Professor Mordahai
 */

/*TODO ====================================================== 
 * C TIMER - Compare each of the following to CPU
 * * CPU
 * * GPU FIND DISTANCES/CPU LABEL AND UPDATE MEANS
 * * * Thread per Vector
 * * * Thread per Centroid
 * * * Thread per Centroid: Centroid Loaded Copied into Constant
 * * * ??? Possible load  max shared memory
 * * * * Group COmmunication sync over multiple blocks
 * * CPU FIND DISTANCES/GPU LABEL MINS/CPU UPDATE MEANS
 * * * Thread per Vector
 * * * Thread per Vector: Combine Into FASTEST Find Distance Kernel
 * * * * Stream multiple kernels at once
 * * CPU FIND DISTANCES/CPU LABEL MINS/GPU UPDATE MEANS
 * * * Atomic Locks to update means
 * * * !Know label size, sum reduction and cpu divides by size
 * * * * Thrust Reduction
 * Pick fastest kernel of the sections and combine them
 * Then work on possible speed ups per kernel
 *
 * Streams?Mutual groups?
 ****!RUN MULTIPLE KS in MULTIPLE DIMENSIONS ON THIS!****
 */// ======================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fcntl.h>
#include "string.h"
#include <kernels.cu>
#include <launches.cu>

#define DEFAULT_ITERATION  100
#define DEFAULT_DIMENSION 50
#define DEFAULT_FILENAME "glove.6B.50d.txt"
#define DEFAULT_N_WORDS 400000
#define DEFAULT_K 20

double *read( char *filename, int dim, int n){
  
  if ( !filename || filename[0] == '\0') {
    fprintf(stderr, "no file name\n");
    return NULL;  // fail
  }

  FILE *fp;

  fprintf(stderr, "read( %s )\n", filename);
  fp = fopen( filename, "rb");
  if (!fp) 
    {
      fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
      return NULL; // fail 
    }
  
  double *data = (double *)malloc( n * dim * sizeof(double));
  if (!data) {
    fprintf(stderr, "read()  unable to allocate %d x %d unsigned ints\n", n, dim);
    return NULL; // fail but return
  }

  char * line = NULL;
  size_t len = 0;
  const char s[3] = " \n";

  int i = 0;
  while ((getline(&line, &len, fp)) != -1) {
    //printf("Retrieved line of length %zu:\n", read);
    //printf("%s", line);
    char *token;
    /* get the first token */
    token = strtok(line, s);
    //data[i] = data[i]=atof(token);
    //i++;
    /* walk through other tokens */
    while( (token = strtok(NULL, s)) != NULL ) {
      data[i]=atof(token);
      //printf( "%f\n", data[i]);
      i++;
      //printf( " %s\n", token);
    }
  }
  fclose(fp);
  if (line)
    free(line);
  return data; // success
}

int main( int argc, char **argv ){
  int n = DEFAULT_N_WORDS;
  int k = DEFAULT_K;
  int iter = DEFAULT_ITERATION;
  int dim = DEFAULT_DIMENSION;
  char *filename;
  filename = strdup(DEFAULT_FILENAME);

  int *labelsGOLD,*labelsGPU, *sizes;
  double *data,* centroidsCPU, * centroidsSTD,* centroidsGPU,* distances, * mins, *maxs;
  time_t CPUstart, CPUend, GPUstart, GPUend;
  
  if (argc > 1) {
    if (argc == 6)  {
      filename = strdup( argv[1]);
      k = atoi( argv[2] );
      iter = atoi( argv[3] );
      n = atoi( argv[4] );
      dim = atoi( argv[5] );
    }
    if (argc == 3) {
      k = atoi( argv[1] );
      iter = atoi( argv[2]);
    }
  }
  fprintf(stderr, "file %s\nk %d\niter %d\ndim %d\nn %d\n", filename, k, iter, dim, n); 

  //Read Data and Init Centroids ===========================================
  data = read(filename, dim, n);

  //Find Mins and Maxs
  mins = (double*)malloc(dim * sizeof(double));
  if (!mins) {
    fprintf(stderr, "mins  unable to allocate %d doubles\n", dim);
    return -1; // fail but return
  }
  maxs = (double*)malloc(dim * sizeof(double));
  if (!maxs) {
    fprintf(stderr, "maxss unable to allocate %d doubles\n", dim);
    return -1; // fail but return
  }
  //Init Centroids
  centroidsSTD = (double*)malloc(k*dim*sizeof(double));
  if (!centroidsSTD) {
    fprintf(stderr, "centroids  unable to allocate %d x %d doubles\n", k, dim);
    return -1; // fail but return
  }
  for(int i = 0; i < dim; i++){
    double min = 9999999;
    double max = -9999999;
    for(int vector = 0+i; vector < n*dim; vector += dim){
      if(data[vector] < min){
	min = data[vector];
      }
      if(data[vector] > max){
	max = data[vector];
      }
    }
    mins[i] = min;
    maxs[i] = max;
    //fprintf(stderr, "Dim:%d : Min:%f : Max:%f\n", i, min, max); 
  }
  for(int i = 0; i < k; i++){
    //fprintf(stderr,"Centroid:%d ",i); 
    for(int vector = 0; vector < dim; vector++){
      double scale = rand() / (double) RAND_MAX; /* [0, 1.0] */
      centroidsSTD[(i * dim) + vector] = mins[vector] + scale * (maxs[vector] - mins[vector]);/* [min, max] */
      //fprintf(stderr, "%f/ ", centroidsSTD[(i * dim) + vector]);
    }
    //fprintf(stderr,"\n"); 
  }
  free(mins);
  free(maxs);

  //Allocate Memory
  labelsGPU = (int*)malloc(n * sizeof(int));
  if (!labelsGPU) {
    fprintf(stderr, "labelsGPU unable to allocate %d ints\n", n);
    return -1; // fail but return
  }
  // Size of centroid 1, size of centroid 2
  sizes = (int*)malloc(k*sizeof(int));
  if (!sizes) {
    fprintf(stderr, "sizes unable to allocate %d ints\n", dim);
    return -1; // fail but return
  }
  centroidsGPU = (double*)malloc(k * dim * sizeof(double));
  if (!centroidsGPU) {
    fprintf(stderr, "centroidsGPU unable to allocate %d x %d doubles\n", k, dim);
    return -1; // fail but return
  }

  //CPU IMPLEMENTATION =====================================================  
  centroidsCPU = (double*)malloc(k * dim * sizeof(double));
  if (!centroidsCPU) {
    fprintf(stderr, "centroidsCPU  unable to allocate %d x %d doubles\n", k, dim);
    return -1; // fail but return
  }
  distances = (double*)malloc(k * n * sizeof(double));
  if (!centroidsCPU) {
    fprintf(stderr, "distances  unable to allocate %d x %d doubles\n", k, n);
    return -1; // fail but return
  }
  for(int i = 0; i < k*n;i++){
    distances[i]=0;
  }
  for(int i = 0; i < n;i++){
    labelsGPU[i]=-1;
  }
  memcpy(centroidsCPU, centroidsSTD, k * dim * sizeof(double));
  CPUstart=clock();
  labelsGOLD = cpuGOLD(n, k, iter, dim, data, centroidsCPU);
  CPUend=clock();
  free(centroidsCPU);
  fprintf(stderr, "CPU DONE \n");
  fprintf(stderr, "CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);

  //GPU IMPLEMENTATION =====================================================
  // 1a. Thread per Vector **************************************************
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  launchVectorDistance(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU vectorDistance is not equal to CPU GOLD\n"); 
      break;
    }
  }
  fprintf(stderr, "GPU vectorDistance KERNEL is DONE\n");
  fprintf(stderr,"CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr,"GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  // 1b. Thread per Vector UnRolled
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  labelsGPU = launchVectorDistanceUR(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU vectorDistanceUR is not equal to CPU GOLD\n"); 
      break;
    }
  }
  fprintf(stderr, "GPU vectorDistanceUR KERNEL is DONE\n");
  fprintf(stderr,"CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr,"GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  // 2a. Thread per Centroid ***************************************
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  launchCentroidDistance(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU centroidDistance KERNEL is not equal to CPU GOLD\n"); 
      break;
    }
  }
  fprintf(stderr, "GPU centroidDistance KERNEL is done\n");
  fprintf(stderr,"CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr,"GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  // 2b. Thread per Centroid UnRolled
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  launchCentroidDistanceUR(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU centroidDistance Unrolled KERNEL is not equal to CPU GOLD\n"); 
      break;
    }
  }
  fprintf(stderr, "GPU centroidDistance UnRolled KERNEL is done\n");
  fprintf(stderr,"CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr,"GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  // 3a. Thread per Centroid Constant Data ======================================
  
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  launchCentroidConstantDistance(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU centroidConstantDistance is not equal to CPU GOLD\n"); 
      break;
    }
  }
  fprintf(stderr, "GPU centroidConstantDistance KERNEL is done\n");
  fprintf(stderr, "CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr, "GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  
  // 3b. Thread per Centroid Constant Unrolled Data ======================================
  fprintf(stderr, "GPU Thread per Centroid Constant Unrolled KERNEL timing is not faster than previooius solutions\n");
  /*
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  labelsGPU = launchCentroidConstantDistanceUR(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU centroidConstantDistance UNROLLED is not equal to CPU GOLD\n"); 
      break;
    }
  }
  fprintf(stderr, "GPU centroidConstantDistance UNROLLED KERNEL is done\n");
  fprintf(stderr, "CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr, "GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);  
  */
  // 4. labeMins ====================================================================
  fprintf(stderr, "GPU labelMins KERNEL timing is depricated because it only takes off a second or 2\n");
  /*
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  labelsGPU = launchLabelMins(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU labelMins is not equal to CPU GOLD\n"); 
      break;
    }
  }
  fprintf(stderr, "GPU labelMins KERNEL is done\n"); 
  printf("CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  printf("GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  */
  // 5a. Thread per vector with labels **************************************************
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  launchThreadPerVectorWithLabels(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU vectorLabelDistance is not equal to CPU GOLD\n"); 
      //fprintf(stderr, "Gold[%d]=!=GPU[%d]\n",labelsGOLD[i], labelsGPU[i]); 
      break;
    }
  }
  fprintf(stderr, "GPU vectorLabelDistance KERNEL is DONE\n");
  fprintf(stderr,"CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr,"GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  // 5b. Thread per vector unrooled with labels **************************************************
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  launchThreadPerVectorWithLabelsUR(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU vectorLabelDistance UNROLLED is not equal to CPU GOLD\n"); 
      //fprintf(stderr, "Gold[%d]=!=GPU[%d]\n",labelsGOLD[i], labelsGPU[i]); 
      break;
    }
  }
  fprintf(stderr, "GPU vectorLabelDistance UNROLLED KERNEL is DONE\n");
  fprintf(stderr,"CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr,"GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
 
  // 6a. ceontroidConstantDistance + labeMins ===========================================
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  launchCentroidConstantLabels(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU centroidCentroidConstant Distance + Labels is not equal to CPU GOLD\n");
      break;
    }
  }
  fprintf(stderr, "GPU centroidCentroidConstant Distance + Labels KERNEL is done\n");
  fprintf(stderr, "CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr, "GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
   // 6b. ceontroidConstantDistance + labeMins ===========================================
  fprintf(stderr, "GPU centroidConstantData Unrloled + labelMins KERNEL timing is depricated because it is worse tan normal unrolled or constant\n");
  /*
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  labelsGPU = launchCentroidConstantLabelsUR(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,  dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU centroidCentroidConstant Distance + Labels UNROLLED is not equal to CPU GOLD\n");
      break;
    }
  }
  fprintf(stderr, "GPU centroidCentroidConstant Distance + Labels UNROLLED KERNEL is done\n");
  fprintf(stderr, "CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr, "GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  */
  // 7. update Means  ===========================================
  fprintf(stderr, "GPU updateMeans and FULLPGUvector and centroid, atomics suckass, KERNEL timing is depricated because it only takes 10 percent off timing\n");
  /*
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  labelsGPU =  launchUpdateMeans(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU updateMeans is not equal to CPU GOLD\n");
      //fprintf(stderr, " at %d\n", i); 
      break;
    }
  }
  fprintf(stderr, "GPU updateMeans KERNEL is done\n");
  fprintf(stderr, "CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr, "GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  
  //8. FULL GPU vectorLabel Implementation ==================================
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  launchGPUVector(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    //fprintf(stderr, "Vector:%d Gold:%d = labelsGPU:%d\n", i,labelsGOLD[i],labelsGPU[i] );
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU FULLGPU vector is not equal to CPU GOLD\n");
      fprintf(stderr, " at %d\n", i); 
      break;
    }
  }
  fprintf(stderr, "GPU FULLGPU using Vector Label KERNEL is done\n");
  fprintf(stderr, "CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr, "GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  //9. FULL GPU ConstantCentroidLabel Implementation ==================================
  memcpy(centroidsGPU, centroidsSTD, k * dim * sizeof(double));
  GPUstart = clock();
  launchGPUCentroid(labelsGPU, data, distances, centroidsGPU, sizes, n, k, iter,dim);
  GPUend = clock();
  //CORRECTNESS 
  for(int i = 0; i < n; i++){
    //fprintf(stderr, "Vector:%d Gold:%d = labelsGPU:%d\n", i,labelsGOLD[i],labelsGPU[i] );
    if(labelsGOLD[i] != labelsGPU[i]){
      fprintf(stderr, "GPU FULLGPU using Constant Centroid is not equal to CPU GOLD\n");
      fprintf(stderr, " at %d\n", i); 
      break;
    }
  }
  fprintf(stderr, "GPU FULLGPU using Constant Centroid KERNEL is done\n");
  fprintf(stderr, "CPU:TIME %f\n", (double) (CPUend-CPUstart)/CLOCKS_PER_SEC);
  fprintf(stderr, "GPU:TIME %f\n", (double) (GPUend-GPUstart)/CLOCKS_PER_SEC);
  */
  //10. GPU with full Reduction ================================
  fprintf(stderr, "GPU FULL REDUCTION KERNEL timing is depricated because its not implemented\n");
  
  fprintf(stderr, "GPU DONE \n"); 

  free(labelsGOLD);
  free(labelsGPU);
  free(sizes);
  free(centroidsGPU);
  free(centroidsSTD);
  free(data);
}
