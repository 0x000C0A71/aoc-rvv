#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>





// memcopy knockoff
void memclone(int8_t *restrict from, int8_t *restrict to, size_t length) {
	while (length > 0) {
		size_t k = __riscv_vsetvl_e32m8(length);

		vint8m8_t transit = __riscv_vle8_v_i8m8(from, k);
		__riscv_vse8_v_i8m8(to, transit, k);

		from += k;
		to += k;
		length -= k;
	}
}


typedef struct {
	size_t capacity;
	size_t size;
	int8_t* target;
} Smplvc;

void Smplvc_add(Smplvc* vc, int8_t value) {
	if (vc->size >= vc->capacity) {

		// realloc
		size_t new_cap = vc->size << 1;

		int8_t* new = malloc(sizeof(int8_t)*new_cap);
		memclone(vc->target, new, vc->size);
		free(vc->target);

		vc->capacity = new_cap;
		vc->target = new;
	}

	vc->target[vc->size++] = value;
}

/** This is basically what we're doing vectorized
typedef int bool;
#define true 1
#define false 0

bool isFit(int8_t* buff, size_t stride) {
	int8_t* local_buffer = buff;

	int8_t prev = *local_buffer;

	bool couldBeIncreasing = true;
	bool couldBeDecreasing = true;
	bool areLevelsGood = true;

	bool isFinished = false;

	for (int i = 1; i < stride; i++) {
		local_buffer++;

		int8_t this = *local_buffer;

		isFinished |= this == 0;

		couldBeIncreasing &= (this > prev) | isFinished;
		couldBeDecreasing &= (this < prev) | isFinished;

		int8_t absdiff = abs(this - prev);

		areLevelsGood &= (absdiff <= 3) | isFinished;
		prev = this;
	}

	return areLevelsGood & (couldBeIncreasing | couldBeDecreasing);
}
*/


uint64_t work_work2(int8_t* buffer, size_t stride, size_t n) {

	printf("Will process %d lines of up to %d values each...\n", n, stride);
	
	size_t iterations = 0;

	uint64_t total = 0;

	while (n > 0) {
		const size_t k = __riscv_vsetvl_e8m4(n);

		int8_t* local_buffer = buffer;

		vint8m4_t prev = __riscv_vlse8_v_i8m4(local_buffer, stride, k);


		vbool2_t could_be_increasing = __riscv_vmset_m_b2(k);
		vbool2_t could_be_decreasing = __riscv_vmset_m_b2(k);
		vbool2_t are_levels_good     = __riscv_vmset_m_b2(k);

		vbool2_t is_finished         = __riscv_vmclr_m_b2(k);

		for (int i = 1; i < stride; i++) {
			local_buffer++;

			const vint8m4_t this = __riscv_vlse8_v_i8m4(local_buffer, stride, k);

			const vbool2_t is_zero = __riscv_vmseq_vx_i8m4_b2(this, 0, k);
			is_finished = __riscv_vmor_mm_b2(is_zero, is_finished, k);

			const vbool2_t is_greater = __riscv_vmsgt_vv_i8m4_b2(this, prev, k);
			const vbool2_t can_increase = __riscv_vmor_mm_b2(is_greater, is_finished, k); // could be achieved with masking
			could_be_increasing = __riscv_vmand_mm_b2(can_increase, could_be_increasing, k);

			const vbool2_t is_lesser = __riscv_vmslt_vv_i8m4_b2(this, prev, k);
			const vbool2_t can_decrease = __riscv_vmor_mm_b2(is_lesser, is_finished, k); // could be achieved with masking
			could_be_decreasing = __riscv_vmand_mm_b2(can_decrease, could_be_decreasing, k);

			const vint8m4_t diff1 = __riscv_vsub_vv_i8m4(this, prev, k);
			const vint8m4_t diff2 = __riscv_vneg_v_i8m4(diff1, k);
			const vint8m4_t absdiff = __riscv_vmax_vv_i8m4(diff1, diff2, k);

			const vbool2_t is_level_good = __riscv_vmsle_vx_i8m4_b2(absdiff, 3, k);
			const vbool2_t can_manage = __riscv_vmor_mm_b2(is_level_good, is_finished, k); // could be achieved with masking
			are_levels_good = __riscv_vmand_mm_b2(can_manage, are_levels_good, k);

			prev = this;
		}

		const vbool2_t inc_dec_good = __riscv_vmor_mm_b2(could_be_increasing, could_be_decreasing, k);
		const vbool2_t final_flags = __riscv_vmand_mm_b2(are_levels_good, inc_dec_good, k);
		
		total += __riscv_vcpop_m_b2(final_flags, k);

		n -= k;
		buffer += stride*k;
		iterations++;
	}

	printf("Processed in %d iterations!\n", iterations);

	return total;
}


int main() {
	Smplvc vec = {
		.capacity = 1,
		.size = 0,
		.target = malloc(sizeof(uint32_t)),
	};
	
	// parsing input
	int32_t curr = 0;
	int counter = 7;
	while (1) {
		const uint8_t c = getchar();

		if (c <= '9' && c >= '0') {
			const uint8_t digit = c - '0';
			curr = curr*10 + digit;
		} else if (c == '\n' || c == 0xff) {
			if (curr != 0) {
				Smplvc_add(&vec, curr);
				curr = 0;
			}
			for (; counter > 0; counter--) Smplvc_add(&vec, 0);
			counter = 7;
		} else if (curr != 0) {
			Smplvc_add(&vec, curr);
			curr = 0;
			counter--;
		}

		if (c == 0xff) break;
	}

	int samples = vec.size/8;

	int result = work_work2(vec.target, 8, samples);

	printf("got %d\n", result);
}
