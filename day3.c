#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <riscv_vector.h>
#include <stdlib.h>

// memcpy knockoff
void memclone(uint16_t *restrict from, uint16_t *restrict to, size_t length) {
	while (length > 0) {
		size_t k = __riscv_vsetvl_e16m8(length);

		vuint16m8_t transit = __riscv_vle16_v_u16m8(from, k);
		__riscv_vse16_v_u16m8(to, transit, k);

		from += k;
		to += k;
		length -= k;
	}
}

typedef struct {
	size_t capacity;
	size_t size;
	uint16_t* target;
} Smplvc;

void Smplvc_add(Smplvc* vc, uint16_t value) {
	if (vc->size >= vc->capacity) {

		// realloc
		size_t new_cap = vc->size << 1;

		uint16_t* new = malloc(sizeof(uint16_t)*new_cap);
		memclone(vc->target, new, vc->size);
		free(vc->target);

		vc->capacity = new_cap;
		vc->target = new;
	}

	vc->target[vc->size++] = value;
}


typedef struct {
	uint16_t a, b;
	bool is_valid;
} ParseResult;


typedef struct {
	uint8_t next;
	uint32_t pos;
} ParseHead;


uint8_t ParseHead_next(ParseHead* head) {
	const uint8_t old = head->next;
	head->next = getchar();
	head->pos++;
	return old;
}

bool ParseHead_expect(ParseHead* head, uint8_t is) {
	if (head->next != is) return false;
	ParseHead_next(head);
	return true;
}

ParseResult parse_one(ParseHead* head) {
	const ParseResult null = { .is_valid = false };

	if (!ParseHead_expect(head, 'm')) return null;
	if (!ParseHead_expect(head, 'u')) return null;
	if (!ParseHead_expect(head, 'l')) return null;
	if (!ParseHead_expect(head, '(')) return null;
	
	uint16_t nums[2] = {};
	const uint8_t delimiters[2] = ",)";

	for (int j = 0; j < 2; j++) {
		uint16_t num = 0;
		for (int i = 0; i < 3; i++) {
			const uint8_t next = head->next;

			if (next > '9' || next < '0') break;

			num = num*10 + (uint16_t)(next - '0');
			ParseHead_next(head);
		}
		if (num == 0) return null;

		nums[j] = num;

		if (!ParseHead_expect(head, delimiters[j])) return null;
	}

	return (ParseResult){
		.a = nums[0],
		.b = nums[1],
		.is_valid = true,
	};
}




uint32_t dot_product(uint16_t *restrict a, uint16_t *restrict b, size_t n) {
	
	printf("Will calculate dot product of 2 vectors fo %d elements each...\n", n);
	
	vuint32m1_t acc = __riscv_vmv_s_x_u32m1(0, 1);
	size_t iterations = 0;
	while (n > 0) {
		size_t k = __riscv_vsetvl_e32m8(n);

		const vuint16m4_t as = __riscv_vle16_v_u16m4(a, k);
		const vuint16m4_t bs = __riscv_vle16_v_u16m4(b, k);

		const vuint32m8_t cs = __riscv_vwmulu_vv_u32m8(as, bs, k);

		acc = __riscv_vredsum_vs_u32m8_u32m1(cs, acc, k);

		a += k;
		b += k;
		n -= k;
		iterations++;
	}

	printf("Computation too %d iterations!\n", iterations);

	return __riscv_vmv_x_s_u32m1_u32(acc);
}


uint8_t parse_dodont(ParseHead* head) {
	if (!ParseHead_expect(head, 'd')) return 0;
	if (!ParseHead_expect(head, 'o')) return 0;

	const uint8_t junction = head->next;
	switch (junction) {
		case '(':
			if (!ParseHead_expect(head, '(')) return 0;
			if (!ParseHead_expect(head, ')')) return 0;
			return 1;
		case 'n':
			if (!ParseHead_expect(head, 'n')) return 0;
			if (!ParseHead_expect(head, '\'')) return 0;
			if (!ParseHead_expect(head, 't')) return 0;
			if (!ParseHead_expect(head, '(')) return 0;
			if (!ParseHead_expect(head, ')')) return 0;
			return 2;
		default:
			return 0;
	}
}

int main() {
	Smplvc left = {
		.capacity = 1,
		.size = 0,
		.target = malloc(sizeof(uint16_t)),
	};
	Smplvc right = {
		.capacity = 1,
		.size = 0,
		.target = malloc(sizeof(uint16_t)),
	};


	bool is_enabled = true;
	ParseHead head = {
		.next = getchar(),
		.pos = 0,
	};

	while (head.next != 0xff) {
		uint32_t old_pos = head.pos;

		if (is_enabled) {
			ParseResult res = parse_one(&head);
			if (res.is_valid) {
				Smplvc_add(&left, res.a);
				Smplvc_add(&right, res.b);
				// We could mulitply and accumulate
				// right here inscalar code. This
				// would be WAY faster, but then
				// we wouldn't use ANY rvv
			}
		}
#ifdef PART_B
		const uint8_t dodont = parse_dodont(&head);

		if (dodont == 1) is_enabled = true;
		if (dodont == 2) is_enabled = false;
#endif

		if (old_pos == head.pos) ParseHead_next(&head);
	}

	const uint32_t dp = dot_product(left.target, right.target, left.size);
	printf("got %d\n", dp);
}



