/*************************************************************
*  This file is part of the Surface Evolver source code.     *
*  Programmer:  Ken Brakke, brakke@susqu.edu                 *
*************************************************************/

/* machine.c */

/* missing routines on some systems; see include.h for defines */

#include "include.h"

/**********************************************************************8
*
* function: set_ctypes()
*
* purpose: implement toupper and tolower as arrays and set up
*          ctype flag bits.
*/
void set_ctypes()
{
  int c;
  for ( c = 0xFF ; c > 0 ; c-- )
  { kb_upper_array[c] = (char)c;
    kb_lower_array[c] = (char)c;
  }
  for ( c = 'a' ; c <= 'z' ; c++ )
    kb_upper_array[c] = (char)(c + 'A' - 'a');
  for ( c = 'A' ; c <= 'Z' ; c++ )
    kb_lower_array[c] = (char)(c - 'A' + 'a');

} // end set_ctypes()

/**********************************************************************
*
* function: kb_stricmp()
*
* purpose: case-insensitive string comparison, for systems without
*          stricmp()
*/
int kb_stricmp(char *a, char *b)
{ register char aa,bb;  /* lower case versions of characters */
  for(;;a++,b++)
  { aa = tolower(*a); 
    bb = tolower(*b);
    if ( aa < bb ) return -1;
    if ( aa > bb ) return 1;
    if ( !aa ) break;  /* have reached both nulls */
  }
  return 0;  /* equal strings */
} // end kb_stricmp()

/**********************************************************************
*
* function: kb_strnicmp()
*
* purpose: case-insensitive string comparison, for systems without
*          strnicmp()
*/
int kb_strnicmp(
  char *a,
  char *b,
  int n  /* maximum characters to compare */
)
{ register char aa,bb;  /* lower case versions of characters */
  for(;n;n--,a++,b++)
  { aa = tolower(*a); 
    bb = tolower(*b);
    if ( aa < bb ) return -1;
    if ( aa > bb ) return 1;
    if ( !aa ) break;  /* have reached both nulls */
  }
  return 0;  /* equal strings */
} // end kb_strnicmp()

/**********************************************************************
*
* function: kb_strupr()
*
* purpose: string uppercasing in place, for systems without
*          strupr()
*/
void kb_strupr(char *s)
{
  while ( *s )
     { *s = (char)toupper(*s);
        s++;
     }
} // end kb_strupr()

/**********************************************************************
*
* function: kb_strstr()
*
* purpose: find substring, for systems without
*          strstr()
*/
char *kb_strstr(char *a,char *b)
{
  char *ptr,*ch;

  for ( ; *a ; a++ )
     { for ( ptr = a, ch = b; *ch && (*ptr == *ch) ; ptr++,ch++ ) 
          ;
       if ( *ch == '\0' ) return a;
     }
  return NULL;
} // end kb_strstr()

/**********************************************************************
*
* function: kb_memmove()
*
* purpose: memory move, for systems without memmove()
*/
void kb_memmove(
  char *dest,
  char *src,
  size_t n
)
{
  /* crude bytewise move */
  if ( (dest - src) > 0 )  /* move from top down */
     { 
        src += n; dest += n; 
        for ( ; n ; n-- ) *(--dest) = *(--src);
     }
  else  /* move from bottom up */
     { 
        for ( ; n ; n-- ) *(dest++) = *(src++);
     }
} // end kb_memmove()


