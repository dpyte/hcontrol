#ifndef PIN_LAYOUT_H
#define PIN_LAYOUT_H

#define HCU_PORT_A (*(volatile unsigned char *)0x3)

#ifndef F_CPU
#	define F_CPU 16000000UL
#endif

#endif // PIN_LAYOUT_H
