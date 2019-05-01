#filename = strdup( argv[1]);
#      k = atoi( argv[2] );
#      iter = atoi( argv[3] );
#      n = atoi( argv[4] );
#      dim = atoi( argv[5] );
# Testing dimension
echo "Testing Dimension"
./a.out glove.6B.50d.txt 100 50 400000 50
./a.out glove.6B.100d.txt 100 50 400000 100
./a.out glove.6B.200d.txt 100 50 400000 200
./a.out glove.6B.300d.txt 100 50 400000 300
echo "Testing Size"
./a.out glove.6B.300d.txt 100 50 400000 300
./a.out glove.42B.300d.txt 100 50 1900000 300
./a.out glove.840B.300d.txt 100 50 2200000 300
echo "Testing Centroid"
./a.out glove.840B.300d.txt 10 50 2200000 300
./a.out glove.840B.300d.txt 100 50 2200000 300
./a.out glove.840B.300d.txt 1000 50 2200000 300
./a.out glove.840B.300d.txt 10000 50 2200000 300
echo "Testing ITeration"
./a.out glove.840B.300d.txt 100 10 2200000 300
./a.out glove.840B.300d.txt 100 25 2200000 300
./a.out glove.840B.300d.txt 100 50 2200000 300
./a.out glove.840B.300d.txt 100 100 2200000 300
