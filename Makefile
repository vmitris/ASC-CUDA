CFLAGS += 

a.out: *.cu *.cpp
	nvcc -I$(CUDA_PATH)/samples/common/inc/ $^

clean:
	rm a.out
	
plot:
	gnuplot file.gp

.PHONY: clean
