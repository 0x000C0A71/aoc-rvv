#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>



void strided_qsort(int32_t* array, size_t length) {
	if (length <= 1) return;

	const int32_t p = array[0];
	int32_t* p1 = array + 2;
	int32_t* p2 = array + 2*(length - 1);

	while (p2 >= p1) {
		if (*p1 < p) {
			p1 += 2;
		} else {
			const int32_t tmp = *p1;
			*p1 = *p2;
			*p2 = tmp;
			p2 -= 2;
		}
	}
	*array = *p2;
	*p2 = p;

	const size_t left_count  = (p2-array) >> 1;
	const size_t right_count = length - left_count - 1;

	strided_qsort(array, left_count );
	strided_qsort(p1   , right_count);
}




// memcopy knockoff
void memclone(int32_t *restrict from, int32_t *restrict to, size_t length) {
	while (length > 0) {
		size_t k = __riscv_vsetvl_e32m8(length);

		vint32m8_t transit = __riscv_vle32_v_i32m8(from, k);
		__riscv_vse32_v_i32m8(to, transit, k);

		from += k;
		to += k;
		length -= k;
	}
}


typedef struct {
	size_t capacity;
	size_t size;
	int32_t* target;
} Smplvc;

void Smplvc_add(Smplvc* vc, int32_t value) {
	if (vc->size >= vc->capacity) {

		// realloc
		size_t new_cap = vc->size << 1;

		int32_t* new = malloc(sizeof(int32_t)*new_cap);
		memclone(vc->target, new, vc->size);
		free(vc->target);

		vc->capacity = new_cap;
		vc->target = new;
	}

	vc->target[vc->size++] = value;
}

int32_t main_work(int32_t* buffer, size_t n) {
	printf("Will crunch %ld numbers...\n", n);

	vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, 1);
	size_t iterations = 0;

	while (n > 0) {
		const size_t k = __riscv_vsetvl_e32m4(n);

		const vint32m4x2_t ld1  = __riscv_vlseg2e32_v_i32m4x2(buffer, k);
		const vint32m4_t   a    = __riscv_vget_v_i32m4x2_i32m4(ld1, 0);
		const vint32m4_t   b    = __riscv_vget_v_i32m4x2_i32m4(ld1, 1);
		const vint32m4_t   sd   = __riscv_vsub_vv_i32m4(a, b, k);
		const vint32m4_t   neg  = __riscv_vneg_v_i32m4(sd, k);
		const vint32m4_t   diff = __riscv_vmax_vv_i32m4(sd, neg, k);
		acc = __riscv_vredsum_vs_i32m4_i32m1(diff, acc, k);

		n -= k;
		buffer += 2*k;
		iterations++;
	}

	printf("Finished all the work in just %ld iterations!\n", iterations);

	return __riscv_vmv_x_s_i32m1_i32(acc);
}


int main() {
	Smplvc vec = {
		.capacity = 1,
		.size = 0,
		.target = malloc(sizeof(uint32_t)),
	};
	
	// parsing input
	int32_t curr = 0;
	while (1) {
		const uint8_t c = getchar();

		if (c <= '9' && c >= '0') {
			const uint8_t digit = c - '0';
			curr = curr*10 + digit;
		} else if (curr != 0) {
			Smplvc_add(&vec, curr);
			curr = 0;
		}

		if (c == 0xff) break;
	}

	size_t n = vec.size >> 1;

	strided_qsort(vec.target  , n);
	strided_qsort(vec.target+1, n);

	int32_t result = main_work(vec.target, n);

	printf("total %d\n", result);

}
