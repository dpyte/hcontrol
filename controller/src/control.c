//  It has to be on top of the include macros
#include "pin_layout.h"

#include <avr/io.h>
#include <util/delay.h>


// Pin A0 - 5v output

int main(void) {
	DDRC = 0xff;
	for (;;) {
		PORTC = 0xff;
		_delay_ms(1000);
		PORTC = 0x00;
		_delay_ms(1000);
	}
	return 0;
}
