#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


// memcopy knockoff
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




// memset knockoff
void memfill(uint8_t *restrict target, size_t size, uint8_t value) {

	while (size > 0) {

		size_t k = __riscv_vsetvl_e8m8(size);
		const vuint8m8_t ones = __riscv_vmv_v_x_u8m8(value, k);
		__riscv_vse8_v_u8m8(target, ones, k);

		size -= k;
		target += k;
	}
}

void fill_map(uint8_t *restrict invalid_map, uint8_t *restrict input, size_t lines) {
	
	while (lines > 0) {

		size_t k = __riscv_vsetvl_e8m4(lines);

		const vuint8m4x2_t pack = __riscv_vlseg2e8_v_u8m4x2(input, k);
		const vuint8m4_t left  = __riscv_vget_v_u8m4x2_u8m4(pack, 0);
		const vuint8m4_t right = __riscv_vget_v_u8m4x2_u8m4(pack, 1);

		const vuint16m8_t hunnie = __riscv_vwmulu_vx_u16m8(left, 100, k);
		const vuint16m8_t fattie = __riscv_vzext_vf2_u16m8(right, k);
		const vuint16m8_t offset = __riscv_vadd_vv_u16m8(hunnie, fattie, k);

		const vuint8m4_t ones = __riscv_vmv_v_x_u8m4(1, k);
		__riscv_vsuxei16_v_u8m4(invalid_map, offset, ones, k);

		lines -= k;
		input += 2*k;
	}
}


/** This is basically what we are doing in SIMD
#include <stdbool.h>

uint8_t check_line(uint8_t *restrict invalid_map, uint8_t *restrict line, size_t line_length) {
	uint8_t* half_line = line;
	uint8_t* this_line = line + 1;
	uint8_t* prev_line = line;

	bool inc_half = true;

	bool is_busy = true;
	uint8_t result = 0;

	while (is_busy) {
		const uint8_t this = *this_line;

		const bool is_zero = this == 0 && is_busy;
		is_busy &= !is_zero;
		result = is_zero ? *half_line : result;

		const uint8_t prev = *prev_line;
		const uint8_t val = invalid_map[prev + 100*this];
		const bool brek = val && is_busy;

		is_busy &= !brek;




		this_line++;
		prev_line++;
		if (inc_half) half_line++;
		inc_half = !inc_half;
	}
	return result;
}
*/

uint16_t check_lines(uint8_t *restrict invalid_map, uint8_t* line, size_t line_length, size_t line_count) {
	
	printf("will work on %d lines...\n", line_count);

	vuint16m1_t acc = __riscv_vmv_s_x_u16m1(0, 1);

	size_t iterations = 0;

	while (line_count > 0) {
		uint8_t* half_line = line;
		uint8_t* this_line = line + 1;
		uint8_t* prev_line = line;

		size_t k = __riscv_vsetvl_e8m4(line_count);

		int inc_half = 1;

		vbool2_t is_busy = __riscv_vmset_m_b2(k);
		vuint8m4_t result = __riscv_vmv_v_x_u8m4(0, k);

		while (__riscv_vcpop_m_b2(is_busy, k) != 0) {
			const vuint8m4_t this = __riscv_vlse8_v_u8m4(this_line, line_length, k);

			const vbool2_t is_zero = __riscv_vmseq_vx_u8m4_b2(this, 0, k);
			const vbool2_t should_update = __riscv_vmand_mm_b2(is_zero, is_busy, k);
			const vbool2_t updator = __riscv_vmnot_m_b2(should_update, k);

			is_busy = __riscv_vmand_mm_b2(is_busy, updator, k);
			const vuint8m4_t half = __riscv_vlse8_v_u8m4(half_line, line_length, k);
			result = __riscv_vmerge_vvm_u8m4(result, half, should_update, k);
			//result = __riscv_vmerge_vvm_u8m1(half, result, updator, k);


			const vuint8m4_t prev = __riscv_vlse8_v_u8m4(prev_line, line_length, k);
			const vuint16m8_t hunnie = __riscv_vwmulu_vx_u16m8(this, 100, k);
			const vuint16m8_t fattie = __riscv_vzext_vf2_u16m8(prev, k);
			const vuint16m8_t offset = __riscv_vadd_vv_u16m8(hunnie, fattie, k);
			const vuint8m4_t val = __riscv_vluxei16_v_u8m4(invalid_map, offset, k);
			const vbool2_t booled = __riscv_vmsne_vx_u8m4_b2(val, 0, k);
			const vbool2_t brek = __riscv_vmand_mm_b2(booled, is_busy, k);
			const vbool2_t brekor = __riscv_vmnot_m_b2(brek, k);
			is_busy = __riscv_vmand_mm_b2(is_busy, brekor, k);



			this_line++;
			prev_line++;
			if (inc_half) half_line++;
			inc_half = !inc_half;
		}

		const vuint16m8_t ext = __riscv_vzext_vf2_u16m8(result, k);
		acc = __riscv_vredsum_vs_u16m8_u16m1(ext, acc, k);

		line += k*line_length;
		line_count -= k;
		iterations++;
	}
	
	printf("Needed %d iterations!\n", iterations);

	return __riscv_vmv_x_s_u16m1_u16(acc);
}




int main() {
	Smplvc vec = {
		.capacity = 1,
		.size = 0,
		.target = malloc(sizeof(uint8_t)),
	};


	// parse invalidities
	while (1) {
		const uint8_t indicator = getchar();

		if (indicator == '\n') break;

		const uint8_t c1 = indicator - '0';
		const uint8_t c2 = getchar() - '0';
		getchar();
		const uint8_t c3 = getchar() - '0';
		const uint8_t c4 = getchar() - '0';
		getchar();

		Smplvc_add(&vec, (c1*10 + c2));
		Smplvc_add(&vec, (c3*10 + c4));
	}

	
	uint8_t* map = malloc(100*100*sizeof(uint8_t));
	memfill(map, 100*100, 0);

	fill_map(map, vec.target, vec.size);

	free(vec.target);
	vec = (Smplvc){
		.capacity = 1,
		.size = 0,
		.target = malloc(sizeof(uint8_t)),
	};


	// parse lines
	const uint8_t line_width = 32;

	size_t line_count = 0;
	uint8_t this_line = line_width;

	while (1) {
		const uint8_t c1 = getchar() - '0';
		const uint8_t c2 = getchar() - '0';
		const uint8_t next = getchar();

		if (next == 0xff) break;

		Smplvc_add(&vec, (c1*10 + c2));
		this_line--;

		if (next == '\n') {
			for (; this_line > 0; this_line--) Smplvc_add(&vec, 0);
			line_count++;
			this_line = line_width;
		}
	}

	// Showing the matrix 
	for (int y = 0; y < 100; y++) {
		for (int x = 0; x < 100; x++) putchar(map[y*100 + x] ? 'X' : '.');
		putchar('\n');
	}
	
	uint16_t res = check_lines(map, vec.target, line_width, line_count);

	printf("got %d\n", res);
}
