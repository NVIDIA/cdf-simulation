
/* to be appended to end of ytab.c so it knows about yytoks */

/******************************************************************
*
*  function: tokname()
*
*  purpose: find name of given token number. Uses yytoks[]
*              list used by debugging.
*
*/


char *tokname(toknum)
int toknum;
{ 
  char *name;

#ifndef NO_YACC_DEBUG
#ifdef YYBISON
  if ( YYTRANSLATE(toknum) == 2 ) /* undefined token */
  { sprintf(errmsg,"Internal error: Token number %d does not have token name in yytname.\n",
       toknum);
    erroutstring(errmsg);
  }

  name = yytname[YYTRANSLATE(toknum)];

  return name;
#else
  int yy_i;

  for ( yy_i = 0; yytoks[yy_i].t_val >= 0; yy_i++ )
  if ( yytoks[yy_i].t_val == toknum )
      return yytoks[yy_i].t_name; 
  /* unfound */
  name = "                         ";
  sprintf(name,"%4d (unnamed)",toknum);
  return name;
#endif
#else
  /* unfound */
  name = "                         ";
  sprintf(name,"%4d (unnamed)",toknum);
  return name;
#endif

}

