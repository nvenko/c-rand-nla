# Copyright (c) 2024 Nicolas Venkovic
#
# This file is part of c-rand-nla.
# 
# This file is licensed under the MIT License.
# For the full license text, see the LICENSE file in the root directory.

# Compiler
CC = gcc

# Path to MKL
MKLROOT = /opt/intel/oneapi/mkl/2024.2

# Compiler and linker flags
CFLAGS = -O2 -DMKL_ILP64 -m64 -I"$(MKLROOT)/include"
LDFLAGS = -m64 -Wl,--start-group \
	  ${MKLROOT}/lib/libmkl_intel_ilp64.a \
	  ${MKLROOT}/lib/libmkl_gnu_thread.a \
	  ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group \
	  -lgomp -lpthread -lm -ldl

# Common source files
COMMON_SRCS = sketching.c fht_avx.c gram_schmidt.c cholesky_qr.c block_gram_schmidt.c misc.c

# Object files for common sources
COMMON_OBJS = $(COMMON_SRCS:.c=.o)

# Example source files
EXAMPLE_SRCS = $(wildcard example*.c)

# Define executables based on example source files
TARGETS = $(EXAMPLE_SRCS:.c=)

# Default rule
all: $(TARGETS)

# Rule to build each example
example%: example%.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Rule to compile the source files
%.o: %.c rand_nla.h
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean the build
clean:
	rm -f $(TARGETS) *.o

# Rule to remove object files but keep the binaries
clean-obj:
	rm -f *.o

# Phony targets
.PHONY: all clean clean-obj