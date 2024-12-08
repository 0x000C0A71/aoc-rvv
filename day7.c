#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>





// memcpy knockoff
void memclone(uint64_t *restrict from, uint64_t *restrict to, size_t length) {
	while (length > 0) {
		size_t k = __riscv_vsetvl_e64m8(length);

		vuint64m8_t transit = __riscv_vle64_v_u64m8(from, k);
		__riscv_vse64_v_u64m8(to, transit, k);

		from += k;
		to += k;
		length -= k;
	}
}


typedef struct {
	size_t capacity;
	size_t size;
	uint64_t* target;
} Smplvc;

void Smplvc_add(Smplvc* vc, uint64_t value) {
	if (vc->size >= vc->capacity) {

		// realloc
		size_t new_cap = vc->size << 1;

		uint64_t* new = malloc(sizeof(uint64_t)*new_cap);
		memclone(vc->target, new, vc->size);
		free(vc->target);

		vc->capacity = new_cap;
		vc->target = new;
	}

	vc->target[vc->size++] = value;
}





int figure_out(uint64_t result, uint64_t* values, size_t val_count) {
	const uint64_t maxperms = 1 << val_count;

	size_t permutations = maxperms;

	size_t iterations = 0;

	int ret = 0;

	while (permutations > 0) {

		size_t k = __riscv_vsetvl_e64m8(permutations);

		permutations -= k;
		iterations++;
		
		const vuint64m8_t ids = __riscv_vid_v_u64m8(k);
		vuint64m8_t perms = __riscv_vadd_vx_u64m8(ids, permutations, k);

		vuint64m8_t last = __riscv_vmv_v_x_u64m8(values[0], k);

		for (int i = 1; i < val_count; i++) {
			const vuint64m8_t lowest_bit = __riscv_vand_vx_u64m8(perms, 1, k);
			perms = __riscv_vsrl_vx_u64m8(perms, 1, k);
			const vbool8_t flag = __riscv_vmsne_vx_u64m8_b8(lowest_bit, 0, k);

			const uint64_t next = values[i];

			const vuint64m8_t mulled = __riscv_vmul_vx_u64m8(last, next, k);

			// These 2 are technically not needed, as the numbers passed to the
			// functione never actually generate an overflow. I've decided to keep
			// it in, however, as I believe it demonstrates an interresting idea
			const vbool8_t mul_offlag = __riscv_vmsltu_vv_u64m8_b8(mulled, last, k);
			const vbool8_t flag2 = __riscv_vmor_mm_b8(flag, mul_offlag, k);

			const vuint64m8_t added  = __riscv_vadd_vx_u64m8(last, next, k);

			// Instead of "disabling" the threads on which an overflow
			// happened (which would honestly be a better idea), I
			// just force these threads to merge the add result here.
			// This way, I just compute this specific add case twice instead
			// of moving some "error value" into the "last" variable
			last = __riscv_vmerge_vvm_u64m8(mulled, added, flag2, k);
		}

		const vbool8_t eq = __riscv_vmseq_vx_u64m8_b8(last, result, k);
		const size_t eqc = __riscv_vcpop_m_b8(eq, k);
		if (eqc != 0) {
			ret = 1;
			break;
		}

	}


	printf("Went through %ld/%ld permutations in %ld iterations (%lu was %s)\n",
		maxperms - permutations,
		maxperms,
		iterations,
		result,
		ret ? "valid" : "invalid"
	);

	return ret;
}



int main() {
	
	uint64_t sum = 0;

	int busy = 1;
	while (busy) {
		uint64_t tot = 0;

		while (1) {
			const uint8_t c = getchar();

			if (c <= '9' && c >= '0') {
				tot = tot*10 + (c - '0');
			} else {
				break;
			}
		}

		Smplvc vec = {
			.capacity = 1,
			.size = 0,
			.target = malloc(sizeof(uint64_t)),
		};

		getchar();

		uint64_t num = 0;
		while (1) {
			const uint8_t c2 = getchar();
			
			if (c2 == 0xff) busy = 0;
			
			if (c2 <= '9' && c2 >= '0') {
				num = num*10 + (c2 - '0');
			} else {
				Smplvc_add(&vec, num);
				num = 0;
			}

			if (c2 == '\n' || c2 == 0xff) break;
		}
		
		if (busy) {
			int val = figure_out(tot, vec.target, vec.size);
			if (val) sum += tot;
		}
		
		free(vec.target);
	}

	printf("got %lu\n", sum);

}

