/*************************************************************
*  This file is part of the Surface Evolver source code.     *
*  Programmer:  Ken Brakke, brakke@susqu.edu                 *
*************************************************************/

/************************************************************
*
*  File:  lex.h
*
*  Contents: Definitions for lexical analyzer for reading
*        data files.
*/

#ifndef _LEX_H_
#define _LEX_H_

#define LEXEME_SIZE 63
/* Data type for yylval and parse semantics stack. */
typedef struct { int i;   /* standard integer yylval */
                 int qnum; /* auxiliary value */
                 int dims; /* number of dimensions for array */
                 REAL r;
                 int etype;  /* geometric element type, or array type */
                 int datatype;  /* expression datatype */ 
                 struct sym *symptr; /* iteration variable names */
                 char lexeme[LEXEME_SIZE+1]; /* for identifiers */
              } yystype;
#define YYSTYPE yystype
extern YYSTYPE yylval;  /* parser terminal value */

#endif

// fixing weird bug with Bison 2.1 that puts in ";" after yyparse()
// function name if __STDC__ not define
#ifdef YYBISON
#ifndef __STDC__
#define __STDC__ 1
#endif
#endif

