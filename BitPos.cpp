#include <string>
#include <iostream>
#include <stdint.h>

class BitPos
{
	public:
		
		BitPos( );
		BitPos( uint8_t* byte );
		BitPos( uint8_t* byte, uint8_t mask );
		~BitPos( );

		void advance(void);

	private:

		uint8_t* byte;
		uint8_t  mask;
};