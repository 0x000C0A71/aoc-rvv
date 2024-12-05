#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>





// memcpy knockoff
void memclone(uint8_t *restrict from, uint8_t *restrict to, size_t length) {
	while (length > 0) {
		size_t k = __riscv_vsetvl_e8m8(length);

		vuint8m8_t transit = __riscv_vle8_v_u8m8(from, k);
		__riscv_vse8_v_u8m8(to, transit, k);

		from += k;
		to += k;
		length -= k;
	}
}


typedef struct {
	size_t capacity;
	size_t size;
	uint8_t* target;
} Smplvc;

void Smplvc_add(Smplvc* vc, uint8_t value) {
	if (vc->size >= vc->capacity) {

		// realloc
		size_t new_cap = vc->size << 1;

		uint8_t* new = malloc(sizeof(uint8_t)*new_cap);
		memclone(vc->target, new, vc->size);
		free(vc->target);

		vc->capacity = new_cap;
		vc->target = new;
	}

	vc->target[vc->size++] = value;
}





uint64_t do_shit_A(uint8_t* buffer, size_t width, size_t height, int xo, int yo) {

	uint8_t* x_buffer = buffer + (xo < 0 ? 3 : 0) + (yo < 0 ? 3 : 0)*width;
	uint8_t* m_buffer = x_buffer + yo*width + xo;
	uint8_t* a_buffer = m_buffer + yo*width + xo;
	uint8_t* s_buffer = a_buffer + yo*width + xo;

	uint64_t total = 0;

	const size_t samples_x = width  - (xo != 0 ? 3 : 0);
	const size_t samples_y = height - (yo != 0 ? 3 : 0);

	int line_its = 0;

	for (size_t y = 0; y < samples_y; y++) {

		line_its = 0;

		size_t its = samples_x;

		uint8_t* lx_buffer = x_buffer;
		uint8_t* lm_buffer = m_buffer;
		uint8_t* la_buffer = a_buffer;
		uint8_t* ls_buffer = s_buffer;


		while (its > 0) {

			size_t k = __riscv_vsetvl_e8m1(its);

			const vuint8m8_t xs = __riscv_vle8_v_u8m8(lx_buffer, k);
			const vbool1_t   xb = __riscv_vmseq_vx_u8m8_b1(xs, 'X', k);
			const vuint8m8_t ms = __riscv_vle8_v_u8m8(lm_buffer, k);
			const vbool1_t   mb = __riscv_vmseq_vx_u8m8_b1(ms, 'M', k);
			const vbool1_t   b1 = __riscv_vmand_mm_b1(xb, mb, k);

			const vuint8m8_t as = __riscv_vle8_v_u8m8(la_buffer, k);
			const vbool1_t   ab = __riscv_vmseq_vx_u8m8_b1(as, 'A', k);
			const vuint8m8_t ss = __riscv_vle8_v_u8m8(ls_buffer, k);
			const vbool1_t   sb = __riscv_vmseq_vx_u8m8_b1(ss, 'S', k);
			const vbool1_t   b2 = __riscv_vmand_mm_b1(ab, sb, k);

			const vbool1_t b3 = __riscv_vmand_mm_b1(b1, b2, k);
			total += __riscv_vcpop_m_b1(b3, k);

			its -= k;
			lx_buffer += k;
			lm_buffer += k;
			la_buffer += k;
			ls_buffer += k;
			line_its++;
		}

		x_buffer += width;
		m_buffer += width;
		a_buffer += width;
		s_buffer += width;

	}

	printf("Did %d line iterations with %d width iterations and found %d!\n", samples_y, line_its, total);

	return total;

}

uint64_t do_shit_B(uint8_t* buffer, size_t width, size_t height) {

	uint8_t* cc_buffer = buffer + 1 + 1*width;
	uint8_t* tl_buffer = cc_buffer - 1 - 1*width;
	uint8_t* tr_buffer = cc_buffer + 1 - 1*width;
	uint8_t* br_buffer = cc_buffer + 1 + 1*width;
	uint8_t* bl_buffer = cc_buffer - 1 + 1*width;

	uint64_t total = 0;

	const size_t samples_x = width  - 2;
	const size_t samples_y = height - 2;

	int line_its = 0;

	for (size_t y = 0; y < samples_y; y++) {

		line_its = 0;

		size_t its = samples_x;

		uint8_t* lcc_buffer = cc_buffer;
		uint8_t* ltl_buffer = tl_buffer;
		uint8_t* ltr_buffer = tr_buffer;
		uint8_t* lbr_buffer = br_buffer;
		uint8_t* lbl_buffer = bl_buffer;


		while (its > 0) {

			size_t k = __riscv_vsetvl_e8m1(its);

			// center
			const vuint8m8_t ccs = __riscv_vle8_v_u8m8(lcc_buffer, k);
			const vbool1_t   ccb = __riscv_vmseq_vx_u8m8_b1(ccs, 'A', k);

			// diagonal 1
			const vuint8m8_t tlv = __riscv_vle8_v_u8m8(ltl_buffer, k);
			const vbool1_t   tlm = __riscv_vmseq_vx_u8m8_b1(tlv, 'M', k);
			const vbool1_t   tls = __riscv_vmseq_vx_u8m8_b1(tlv, 'S', k);
			const vuint8m8_t brv = __riscv_vle8_v_u8m8(lbr_buffer, k);
			const vbool1_t   brm = __riscv_vmseq_vx_u8m8_b1(brv, 'M', k);
			const vbool1_t   brs = __riscv_vmseq_vx_u8m8_b1(brv, 'S', k);
			const vbool1_t   d1a = __riscv_vmand_mm_b1(tlm, brs, k);
			const vbool1_t   d1b = __riscv_vmand_mm_b1(tls, brm, k);
			const vbool1_t   d1  = __riscv_vmor_mm_b1(d1a, d1b, k);

			// diagonal 2
			const vuint8m8_t trv = __riscv_vle8_v_u8m8(ltr_buffer, k);
			const vbool1_t   trm = __riscv_vmseq_vx_u8m8_b1(trv, 'M', k);
			const vbool1_t   trs = __riscv_vmseq_vx_u8m8_b1(trv, 'S', k);
			const vuint8m8_t blv = __riscv_vle8_v_u8m8(lbl_buffer, k);
			const vbool1_t   blm = __riscv_vmseq_vx_u8m8_b1(blv, 'M', k);
			const vbool1_t   bls = __riscv_vmseq_vx_u8m8_b1(blv, 'S', k);
			const vbool1_t   d2a = __riscv_vmand_mm_b1(trm, bls, k);
			const vbool1_t   d2b = __riscv_vmand_mm_b1(trs, blm, k);
			const vbool1_t   d2  = __riscv_vmor_mm_b1(d2a, d2b, k);

			// putting it together
			const vbool1_t ds = __riscv_vmand_mm_b1(d1, d2, k);
			const vbool1_t ft = __riscv_vmand_mm_b1(ds, ccb, k);

			total += __riscv_vcpop_m_b1(ft, k);

			its -= k;
			lcc_buffer += k;
			ltl_buffer += k;
			ltr_buffer += k;
			lbr_buffer += k;
			lbl_buffer += k;
			line_its++;
		}

		cc_buffer += width;
		tl_buffer += width;
		tr_buffer += width;
		br_buffer += width;
		bl_buffer += width;

	}

	printf("Did %d line iterations with %d width iterations and found %d!\n", samples_y, line_its, total);

	return total;

}

int main() {
	Smplvc vec = {
		.capacity = 1,
		.size = 0,
		.target = malloc(sizeof(uint8_t)),
	};

	size_t line_count = 0;
	size_t col_count = 0;
	size_t counter = 0;

	while (1) {
		const uint8_t c = getchar();

		if (c == '\n') {
			col_count = counter;
			counter = 0;
			line_count++;
		} else if (c != 0xff) {
			counter++;
			Smplvc_add(&vec, c);

		}

		if (c == 0xff) break;
	}

	uint64_t total = 0;
#ifdef PART_B
	total += do_shit_B(vec.target, col_count, line_count);
#else
	total += do_shit_A(vec.target, col_count, line_count,  1,  0);
	total += do_shit_A(vec.target, col_count, line_count,  1,  1);
	total += do_shit_A(vec.target, col_count, line_count,  0,  1);
	total += do_shit_A(vec.target, col_count, line_count, -1,  1);
	total += do_shit_A(vec.target, col_count, line_count, -1,  0);
	total += do_shit_A(vec.target, col_count, line_count, -1, -1);
	total += do_shit_A(vec.target, col_count, line_count,  0, -1);
	total += do_shit_A(vec.target, col_count, line_count,  1, -1);
#endif

	printf("got %d\n", total);

}
