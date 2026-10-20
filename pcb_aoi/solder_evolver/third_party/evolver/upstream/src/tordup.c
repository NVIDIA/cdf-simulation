/*************************************************************
*  This file is part of the Surface Evolver source code.     *
*  Programmer:  Ken Brakke, brakke@susqu.edu                 *
*************************************************************/

/***********************************************************************
*
*  File:        tordup.c
*
*  Contents:    tordup() - duplicate torus model in a direction
*
*               detorus() - undo torus mode in accord with display
*                           mode currently in effect.
*
*            
*/

#include "include.h"

static  int v_count;
static  int ecount;
static  int fcount;
static  int bcount;
static  int fecount;

static  element_id  *vlist,*elist,*flist,*felist,*blist;

/***********************************************************************
*
*  Function:   tordup()
*
*  Purpose:    Duplicate a torus configuration  in the direction of
*              one period.  Meant for studying long-term behavior
*              of Ostwald ripening on ever larger scales.  Also good
*              for duplicating a single cell and resolving vertices
*              to try to beat Kelvin.
*
*            
*/
void tordup(int m  /* period to duplicate, 0,1, or 2 */)
{
  vertex_id v_id;
  edge_id    e_id;
  facet_id  f_id;
  body_id    b_id;
  facetedge_id fe_id;
  facetedge_id new_fe;
  facet_id new_f;
  body_id new_b;
  edge_id new_e;
  int i,j;
  REAL *x;
  REAL adjust;
  facetedge_id *body_fe_list;

  v_count = web.skel[VERTEX].max_ord+1;
  ecount = web.skel[EDGE].max_ord+1;
  fcount = web.skel[FACET].max_ord+1;
  bcount = web.skel[BODY].max_ord+1;
  fecount = web.skel[FACETEDGE].max_ord+1;

  /* allocate room for lists of old and new elements */
  vlist = (element_id *)temp_calloc(sizeof(element_id),2*v_count);
  elist = (element_id *)temp_calloc(sizeof(element_id),2*ecount);
  flist = (element_id *)temp_calloc(sizeof(element_id),2*fcount);
  blist = (element_id *)temp_calloc(sizeof(element_id),2*bcount);
  felist = (element_id *)temp_calloc(sizeof(element_id),2*fecount);
  body_fe_list = (element_id *)temp_calloc(sizeof(element_id),2*bcount);
  
  /* record old body data */
  FOR_ALL_BODIES(b_id)
  { blist[loc_ordinal(b_id)] = b_id;
    body_fe_list[loc_ordinal(b_id)] = get_body_fe(b_id);
  }

  /* relocate new vertices */
  FOR_ALL_VERTICES(v_id)
  { /* list old vertices */
    i = loc_ordinal(v_id);
    vlist[i] = v_id;
  }
  for ( i = 0 ; i < v_count ; i++ )
  { /* create corresponding new vertices */
    v_id = vlist[i];
    if ( !valid_id(v_id) ) continue;
    vlist[i+v_count] = dup_vertex(v_id);
    x = get_coord(vlist[i+v_count]);
    for ( j = 0 ; j < SDIM ; j++ )
      x[j] += web.torus_period[m][j];
  }  

  /* fix edge endpoints */
  FOR_ALL_EDGES(e_id)
  { /* list old edges */
    i = loc_ordinal(e_id);
    elist[i] = e_id;
  }
  for ( i = 0 ; i < ecount ; i++ )
  { /* create corresponding new edges */
    vertex_id h,t;
    WRAPTYPE wrap;

    e_id = elist[i];
    if ( !valid_id(e_id) ) continue;
    new_e = dup_edge(e_id);
    elist[i+ecount] = new_e;
    h = get_edge_headv(e_id);
    t = get_edge_tailv(e_id);
    if ( web.modeltype == QUADRATIC )
    { free_element(get_edge_midv(new_e));
      set_edge_midv(new_e,vlist[v_count+loc_ordinal(get_edge_midv(e_id))]);
    }
    wrap = get_edge_wrap(e_id);
    if ( (wrap >> (TWRAPBITS*m)) & WRAPMASK )
    { remove_vertex_edge(h,inverse_id(e_id));
      set_edge_headv(e_id,upgrade(h));
      set_edge_headv(new_e,h);
      set_edge_tailv(new_e,upgrade(t));
    }
    else
    { set_edge_headv(new_e,upgrade(h));
      set_edge_tailv(new_e,upgrade(t));
    }
  }
        
  /* new facets */
  FOR_ALL_FACETS(f_id)
  {
    i = loc_ordinal(f_id);
    flist[i]= f_id;
  }
  for ( i = 0 ; i < fcount ; i++ )
  { /* create corresponding new facets */

    f_id = flist[i];
    if ( !valid_id(f_id) ) continue;
    new_f = dup_facet(f_id);
    flist[i+fcount] = new_f;
     
  }

  /* new facet-edges */
  FOR_ALL_FACETEDGES(fe_id)
  {
    i = loc_ordinal(fe_id);
    felist[i] = fe_id;
  }
  for ( i = 0 ; i < fecount ; i++ )
  { /* create corresponding new facet-edges */

    fe_id = felist[i];
    if ( !valid_id(fe_id) ) continue;
    f_id = get_fe_facet(fe_id);
    e_id = get_fe_edge(fe_id);
    new_fe = new_facetedge(upgrade(f_id),upgrade(e_id));
    felist[i+fecount] = new_fe;
  }

  for ( i = 0 ; i < fecount ; i++ )
  { /* upgrade links */

    fe_id = felist[i];
    if ( !valid_id(fe_id) ) continue;  /* only want old ones */
    new_fe = upgrade(fe_id);
    set_next_edge(new_fe,upgrade(get_next_edge(fe_id)));
    set_prev_edge(new_fe,upgrade(get_prev_edge(fe_id)));
    set_next_facet(new_fe,upgrade(get_next_facet(fe_id)));
    set_prev_facet(new_fe,upgrade(get_prev_facet(fe_id)));
  }
  
  /* upgrade edge fe links */
  for ( i = 0 ; i < ecount ; i++ )
  {
    e_id = elist[i];
    if ( !valid_id(e_id) ) continue;
    fe_id = get_edge_fe(e_id);
    set_edge_fe(upgrade(e_id),upgrade(fe_id));
  }        

  /* go around old facets, linking in new facetedges if necessary */
  for ( i = 0 ; i < fcount ; i++ )
  {
    int newflag;
    facetedge_id fe,first_fe,new_next_fe,next_fe;

    f_id = flist[i];
    if ( !valid_id(f_id) ) continue;
    new_f = upgrade(f_id);

    newflag = 0;  /* whether we should be doing new fe's */
    fe = first_fe = get_facet_fe(f_id);
    if ( !valid_id(fe) ) continue;

    set_facet_fe(new_f,upgrade(fe));

    do
    {
      int sign;

      new_fe = upgrade(fe);
      if ( newflag )
      {
        set_fe_facet(fe,new_f);
        set_fe_facet(new_fe,f_id);
      }
      next_fe = get_next_edge(fe);
      new_next_fe = upgrade(next_fe);
      sign = 0;
      if ( !inverted(get_fe_edge(fe)) )
        if ( get_fe_wrap(fe) & (WRAPMASK<<(TWRAPBITS*m)) )
          sign++;
      if ( inverted(get_fe_edge(next_fe)) )
        if ( get_fe_wrap(next_fe) & (WRAPMASK<<(TWRAPBITS*m)) )
          sign++;
      if ( sign == 1 )
      {
        /* cross-link */
        set_next_edge(fe,new_next_fe);
        set_prev_edge(new_next_fe,fe);
        set_next_edge(new_fe,next_fe);
        set_prev_edge(next_fe,new_fe);
        newflag = !newflag;
      }
      fe = next_fe;
    }
    while ( !equal_id(fe,first_fe) );

    if ( newflag )
    { /* really only one facet, so get rid of new one */
      fe = first_fe;
      do 
      {
        set_fe_facet(fe,f_id);
        fe = get_next_edge(fe);
      }
      while ( !equal_id(fe,first_fe) );
      free_element(new_f);
    }
  }

  /* unwrapping appropriate edges */
  for ( i = 0 ; i < ecount ; i++ )
  { WRAPTYPE mask = WRAPMASK<<(TWRAPBITS*m);
    WRAPTYPE wrap,oldwrap,newwrap;
    int wrapnum;
     
    if ( !valid_id(elist[i]) ) continue;
    wrap = get_edge_wrap(elist[i]); 
    wrapnum = WRAPNUM((wrap>>(TWRAPBITS*m)) & WRAPMASK);
    oldwrap = (wrap & (~mask))|(((wrapnum>>1)&WRAPMASK)<<(TWRAPBITS*m));
    newwrap = (wrap & (~mask))|((((wrapnum+1)>>1)&WRAPMASK)<<(TWRAPBITS*m));
    set_edge_wrap(elist[i],oldwrap);
    set_edge_wrap(elist[i+ecount],newwrap);
  }

  /* create corresponding new facets */
  for ( i = 0 ; i < fcount ; i++ )
  {
    f_id = flist[i];
    if ( !valid_id(f_id) ) continue;
    new_f =  flist[i+fcount];
   
    /* erase body-facet links; will reset later */
    set_facet_body(f_id,NULLBODY);
    set_facet_body(inverse_id(f_id), NULLBODY);
    set_facet_body(new_f,NULLBODY);
    set_facet_body(inverse_id(new_f), NULLBODY);
   
  }
  
  /* new bodies */
  for ( i = 0 ; i < bcount ; i++ )
  { /* create corresponding new facet-edges */

    b_id = blist[i];
    if ( !valid_id(b_id) ) continue;
    new_b = dup_body(b_id);
    blist[i+bcount] = new_b;
    fe_id = body_fe_list[i];
    f_id = get_fe_facet(fe_id);
    set_facet_body(f_id,b_id);
    /* reset, since was wiped earlier */
    body_fe_list[bcount+i] = upgrade(fe_id);
    if ( get_battr(b_id) & WANT_CENTEROFMASS )
    { REAL *oldcm = get_body_cm(b_id);
      REAL *newcm = get_body_cm(new_b);
      oldcm[m] = 0.5;
      newcm[m] = 0.5;
    }
  }

  /* adjust facet bodies */
  if ( web.representation == STRING )
  {
    for ( i = 0 ; i < fcount ; i++ )
      if ( valid_id(flist[i]) )
      { b_id = get_facet_body(flist[i]);
        if ( !valid_id(b_id) ) continue;
        b_id = upgrade(b_id);
        if ( valid_id(flist[i+fcount]) )
           set_facet_body(flist[i+fcount],b_id);
        else free_element(b_id);
      }
  }
  else  /* SOAPFILM */
  { /* have to go around finding contiguous facets of bodies */
    int changeflag;
    facet_id ff_id;

    /* start with canonical facets of old bodies */
    for ( i = 0 ; i < bcount ; i++ )
    {
      b_id = blist[i];
      if ( !valid_id(b_id) ) continue;
      f_id = get_body_facet(b_id);
      set_facet_body(f_id,b_id);
    }
 
    /* go around finding adjacent facets */
    do
    { facetedge_id sentinel;
      changeflag = 0;
      fe_id = NULLFACETEDGE;
      while ( generate_all(FACETEDGE,&fe_id,&sentinel) )
      { if ( equal_id(get_next_facet(fe_id),fe_id) )
          continue; /* valence 1 edge */
        f_id = get_fe_facet(fe_id);
        b_id = get_facet_body(f_id);
        if ( valid_id(b_id) ) 
        { ff_id = inverse_id(get_fe_facet(get_prev_facet(fe_id)));
          if ( !valid_id(get_facet_body(ff_id)) )
          { set_facet_body(ff_id,b_id);
            changeflag++;
          }
        }
        invert(fe_id);
        f_id = get_fe_facet(fe_id);
        b_id = get_facet_body(f_id);
        if ( valid_id(b_id) ) 
        { ff_id = inverse_id(get_fe_facet(get_prev_facet(fe_id)));
          if ( !valid_id(get_facet_body(ff_id)) )
          { set_facet_body(ff_id,b_id);
            changeflag++;
          }
        }
      }
    } while ( changeflag );

    /* now find new bodies whose canonical faces 
       have not been usurped */
    for ( i = 0 ; i < bcount ; i++ )
    {
      b_id = blist[i+bcount];
      if ( !valid_id(b_id) ) continue;

      f_id = get_fe_facet(body_fe_list[i+bcount]);
      if ( valid_id(get_facet_body(f_id)) )
      { /* two bodies are really one */
        free_element(b_id);
        set_body_volume(blist[i],2*get_body_volume(blist[i]),SETSTAMP);
        set_body_oldvolume(blist[i],2*get_body_oldvolume(blist[i]));
        if ( get_battr(b_id) & FIXEDVOL )
          set_body_fixvol(blist[i],2*get_body_fixvol(blist[i]));
        continue;
      } 
      set_facet_body(f_id,b_id);
    }
 
    /* go around finding adjacent facets */
    do
    { facetedge_id sentinel;
      changeflag = 0;
      fe_id = NULLFACETEDGE;
      while ( generate_all(FACETEDGE,&fe_id,&sentinel) )
      { if ( equal_id(get_next_facet(fe_id),fe_id) )
          continue; /* valence 1 edge */
        f_id = get_fe_facet(fe_id);
        b_id = get_facet_body(f_id);
        if ( valid_id(b_id) ) 
        { ff_id = inverse_id(get_fe_facet(get_prev_facet(fe_id)));
          if ( !valid_id(get_facet_body(ff_id)) )
          { set_facet_body(ff_id,b_id);
            changeflag++;
          }
        }
        invert(fe_id);
        f_id = get_fe_facet(fe_id);
        b_id = get_facet_body(f_id);
        if ( valid_id(b_id) ) 
        { ff_id = inverse_id(get_fe_facet(get_prev_facet(fe_id)));
          if ( !valid_id(get_facet_body(ff_id)) )
          { set_facet_body(ff_id,b_id);
            changeflag++;
          }
        }
      }
    } while ( changeflag );
  }
  /* adjust torus period and inverse period matrix */
  for ( i = 0 ; i < SDIM ; i++ )
  { sprintf(msg,"2*(%s)",print_express(&torus_period_expr[m][i],' '));
    cmdptr = msg;
    exparse(0,&torus_period_expr[m][i],USERCOPY);
    cmdptr = NULL;
  }
  if ( web.torus_display_period )
  for ( i = 0 ; i < SDIM ; i++ )
  { sprintf(msg,"2*(%s)",print_express(&torus_display_period_expr[m][i],' '));
    cmdptr = msg;
    exparse(0,&torus_display_period_expr[m][i],USERCOPY);
    cmdptr = NULL;
  }
  calc_periods(NO_ADJUST_VOLUMES);

  /* phase boundary energies */
  if ( phase_flag && (web.representation == STRING) )
  { FOR_ALL_EDGES(e_id)
      set_e_phase_density(e_id);
  }

  if ( phase_flag && (web.representation != STRING) )
  { FOR_ALL_FACETS(f_id)
      set_f_phase_density(f_id);
  }

  if ( everything_quantities_flag )
     reconvert_bodies_to_quantities();

  /* free lists */
  temp_free((char *)vlist);
  temp_free((char *)elist);
  temp_free((char *)flist);
  temp_free((char *)blist);
  temp_free((char *)felist);
  temp_free((char *)body_fe_list);

  /* fix up volconsts */

  calc_content(Q_INFO|Q_FIXED);  /* all volumes */

  /* get volume of piece of unit cell */
  if ( SDIM == 2 )
  {
     adjust = web.torusv  /* /2 */;
  }
  else /* web.representation == SOAPFILM */
  {
     adjust = web.torusv  /* /6 */;
  }

  /* adjust volconsts */
  FOR_ALL_BODIES(b_id)
    { REAL vol = get_body_volume(b_id);
      REAL old = get_body_oldvolume(b_id);
      REAL vc = get_body_volconst(b_id);
      REAL calcvol = vol-vc;
      REAL newvc = old - calcvol;
      newvc = adjust*floor(0.5+newvc/adjust);
      set_body_volconst(b_id,newvc);
      set_body_volume(b_id,calcvol+newvc,SETSTAMP);
   }

  top_timestamp = ++global_timestamp;

} // end tordup()

 
/**************************************************
*
*  Function: upgrade()
*
*  Purpose:  Find the new element corresponding 
*                to an old element.
*/

element_id upgrade(element_id id)
{
    int j = loc_ordinal(id);
    element_id new_id;

    if ( !valid_id(id) ) return NULLID;

    switch( id_type(id) )
      {
         case VERTEX: new_id = vlist[j+v_count];
                      if ( !equal_id(id,vlist[j]) ) invert(new_id);
                      break;
         case EDGE  : new_id = elist[j+ecount];
                      if ( !equal_id(id,elist[j]) ) invert(new_id);
                      break;
         case FACET : new_id = flist[j+fcount];
                      if ( !equal_id(id,flist[j]) ) invert(new_id);
                      break;
         case BODY  : new_id = blist[j+bcount];
                      if ( !equal_id(id,blist[j]) ) invert(new_id);
                      break;
         case FACETEDGE: new_id = felist[j+fecount];
                      if ( !equal_id(id,felist[j]) ) invert(new_id);
                      break;
         default:     new_id = NULLID;
      }
    return new_id;
} // end upgrade()

/********************************************************************
*
* function: reconvert_bodies_to_quantities()
*
* purpose: straightens out body quantities after messing around
*             with rebody() or tordup().  Erases all old instances
*             and recreates whole structure.
*/

void reconvert_bodies_to_quantities()
{ int type;
  int k;
  struct method_instance *mi;
  struct element *e_ptr;
  body_id b_id;
  element_id id;

  for ( type = VERTEX ; type <= FACET ; type++ )
  { int meth_offset = get_meth_offset(type);
    int *methlist;
    FOR_ALL_ELEMENTS(type,id)
    { e_ptr = elptr(id);
      methlist = (int*)((char*)e_ptr+meth_offset);
      for ( k = 0 ; k < (int)e_ptr->method_count ; k++ )
      { 
        mi = METH_INSTANCE(abs(methlist[k]));
        if ( mi->flags & BODY_INSTANCE )
        { methlist[k] = methlist[--(e_ptr->method_count)];
          k--;
        }
      }
    }
  }
  FOR_ALL_BODIES(b_id)
     convert_new_body_to_quantity(b_id);
  convert_bodies_to_quantities();
} // end reconvert_bodies_to_quantities()

             
/*************************************************************************

  detorus() and associated routines for undoing torus mode.

**************************************************************************/

struct dt_vertex {
   vertex_id v_id;  /* final vertex id */
   vertex_id orig_v_id; /* id of source vertex */
   int orig;    /* place in list before sorting */
   int merge_count; /* how many vertices have been merged into this one */
   int newspot;  /* for translating after final sort */
   REAL *x;
} *dt_vlist;
int dt_valloc;
int dt_vcount;
REAL *dt_x_space; // for dt_vertex.x to point at

struct dt_edge {
   edge_id e_id;  /* original */
   int orig;   /* place in list before sorting */
   int *v;  /* edge vertices in dt_vlist */
} *dt_elist;
int dt_ealloc;
int dt_ecount;
int *dt_v_space; // for dt_edge.v to point at

struct dt_facet {
   facet_id f_id;  /* original */
   int e[FACET_EDGES];  /* edges in dt_vllist */
   int *fv; // interior vertices for Lagrange
   int flags;  // for COLORS_SWAPPED indicating orientation
} *dt_flist;
int dt_falloc;
int dt_fcount;
int dt_transform_count; // number of active transforms to use

// prototypes
int *detorus_unify_vertices(void);
int *detorus_unify_edges(void);
void detorus_handle_facet(struct dt_facet *);
void detorus_string_cleanup(void);
void detorus_string_generate(void);
void detorus_soapfilm_cleanup(void);
void detorus_soapfilm_generate(void);

// for debugging, print lists
void detorus_dump()
{ int i;
  printf("\ndetorus vertices\n");
  printf("  N    v_id   orig         x           y           z\n");
  for ( i = 0 ; i < dt_vcount ; i++ )
	printf("%3d   %3s    %3d     %10.8f %10.8f %10.8f\n",i,
           SELNAME(dt_vlist[i].v_id),
 dt_vlist[i].orig,
           (DOUBLE)dt_vlist[i].x[0],(DOUBLE)dt_vlist[i].x[1],
           (DOUBLE)dt_vlist[i].x[2]);
  printf("\ndetorus edges\n");
  printf("  N    e_id   orig     v[0]  v[1]\n");
  for ( i = 1 ; i < dt_ecount ; i++ )
	printf("%3d   %3s    %3d      %3d  %3d\n",i,SELNAME(dt_elist[i].e_id),
	   dt_elist[i].orig,dt_elist[i].v[0],dt_elist[i].v[1]);
  printf("\ndetorus facets\n");
  printf("  N    f_id    e[0] e[1] e[2]\n");
  for ( i = 0 ; i < dt_fcount ; i++ )
	printf("%3d   %3s      %3d  %3d %3d\n",i,SELNAME(dt_flist[i].f_id),
	  dt_flist[i].e[0], dt_flist[i].e[1],dt_flist[i].e[2]);
  printf("\n\n");

}

REAL dt_eps = DT_EPS_DEFAULT;
REAL dt_random_vector[MAXCOORD] = { 1.134515113434,0.7325367367352134,0.136134146732244,
    1.03998837245456};

/*********************************************************************
* function: dt_vertex_key()
* purpose: ordering value for sorting
*/
REAL dt_vertex_key(struct dt_vertex *a)
{ return dot(a->x,dt_random_vector,SDIM);
}

/**********************************************************************
* function: dt_vertex_comp()
* purpose: Comparison function for sorting along random direction
*/
int dt_vertex_comp(struct dt_vertex *a,struct dt_vertex *b)
{ 
  REAL akey,bkey;

  akey = dt_vertex_key(a);
  bkey = dt_vertex_key(b);

  if ( akey < bkey ) return -1;
  if ( akey > bkey ) return  1;
  
  return 0;
} // end dt_vertex_comp()

/**********************************************************************
* function: dt_vid_comp()
* purpose: Comparison function for sorting kept vertices by original v_id
*/
int dt_vid_comp(struct dt_vertex *a,struct dt_vertex *b)
{ 
  if ( a->orig_v_id < b->orig_v_id ) return -1;
  if ( a->orig_v_id > b->orig_v_id ) return  1;
  
  return dt_vertex_comp(a,b);
} // end dt_vid_comp()

/***********************************************************************
* function: dt_vertex_diff()
* purpose: determine if two vertices should be merged
*/
// Return values:
#define DT_DIFF_IN_RANGE 0
#define DT_MERGEABLE 1
#define DT_OUT_OF_RANGE 2
int dt_vertex_diff(struct dt_vertex *a,struct dt_vertex *b)
{ int i;
  REAL akey,bkey;

  akey = dt_vertex_key(a);
  bkey = dt_vertex_key(b);

  if ( fabs(akey-bkey) > 2*dt_eps )
    return DT_OUT_OF_RANGE;

  // Compare parent vertices only if both have them.
  if ( !detorus_sticky )
  { if ( valid_id(a->v_id) && valid_id(b->v_id) )
    { if ( a->v_id != b->v_id ) return DT_DIFF_IN_RANGE;
    }
  }

  for ( i = 0 ; i < SDIM; i++ )
  { if ( a->x[i] < b->x[i] - dt_eps ) return DT_DIFF_IN_RANGE;
    if ( a->x[i] > b->x[i] + dt_eps ) return DT_DIFF_IN_RANGE;
  }
  return DT_MERGEABLE;
} // end dt_vertex_diff()

/**************************************************************************
* function: dt_edge_comp()
* purpose: comparison for sorting edges
*/
int dt_edge_comp(struct dt_edge *a,struct dt_edge *b)
{ int i;
  /* sort by low vertex number, high vertex number, regardless of head/tail */
  for ( i = 0 ; i < EDGE_VERTS ; i++ )
    if ( (a->v[i] < b->v[0]) && (a->v[i] < b->v[1]) ) return -1;
  for ( i = 0 ; i < EDGE_VERTS ; i++ )
    if ( (b->v[i] < a->v[0]) && (b->v[i] < a->v[1]) ) return  1;
  for ( i = 0 ; i < EDGE_VERTS ; i++ )
    if ( (a->v[i] > b->v[0]) && (a->v[i] > b->v[1]) ) return  1;
  for ( i = 0 ; i < EDGE_VERTS ; i++ )
    if ( (b->v[i] > a->v[0]) && (b->v[i] > a->v[1]) ) return -1;
   
  return 0;
} // end dt_edge_comp()

/*************************************************************************
* function: dt_compare()
* purpose: comparison function for sorting facets
*/
int dt_fcompare(struct dt_facet *a,struct dt_facet *b)
{ int i;
  int aa[FACET_EDGES],bb[FACET_EDGES];

  if ( !detorus_sticky )
    if ( !equal_element(a->f_id,b->f_id) )
    { if ( loc_ordinal(a->f_id) < loc_ordinal(b->f_id) ) return -1;
      if ( loc_ordinal(a->f_id) > loc_ordinal(b->f_id) ) return  1;
    }

  // compare absolute values of edge numbers in numerical order
  for ( i = 0 ; i < FACET_EDGES ; i++ )
  { aa[i] = abs(a->e[i]);
    bb[i] = abs(b->e[i]);
  }
  // bubble sort
  if ( aa[1] < aa[0] ) { int tmp = aa[1]; aa[1] = aa[0]; aa[0] = tmp; }
  if ( aa[2] < aa[1] ) { int tmp = aa[2]; aa[2] = aa[1]; aa[1] = tmp; }
  if ( aa[1] < aa[0] ) { int tmp = aa[1]; aa[1] = aa[0]; aa[0] = tmp; }
  if ( bb[1] < bb[0] ) { int tmp = bb[1]; bb[1] = bb[0]; bb[0] = tmp; }
  if ( bb[2] < bb[1] ) { int tmp = bb[2]; bb[2] = bb[1]; bb[1] = tmp; }
  if ( bb[1] < bb[0] ) { int tmp = bb[1]; bb[1] = bb[0]; bb[0] = tmp; }

  for ( i = 0 ; i < FACET_EDGES ; i++ )
  { if ( aa[i] < bb[i] ) return -1;
    if ( aa[i] > bb[i] ) return  1;
  }

  return 0;
} // end dt_fcompare()


/**************************************************************************
*
* function: detorus()
* purpose: Implementation of detorus command to expand symmetries 
*/
void detorus() 
{ int old_box_flag = box_flag;

  if ( web.representation == SIMPLEX )
    kb_error(5396,"detorus not doing simplex model yet.\n",RECOVERABLE);
  if ( web.torus_flag)
  { if ( web.modeltype == QUADRATIC )
      kb_error(5397,"detorus not doing quadratic torus model yet.\n",RECOVERABLE);
    if ( web.modeltype == LAGRANGE )
      kb_error(5398,"detorus not doing Lagrange torus model yet.\n",RECOVERABLE);
  }
  
  box_flag = 0; // don't want box edges 
  dt_transform_count = (transforms_flag ? transform_count : 1);

  if ( !web.torus_flag )
  { if ( web.representation == STRING )
      detorus_string_generate();
    else if (web.representation == SOAPFILM )
      detorus_soapfilm_generate();
  }
  else
     /* Use graphics generator to generate facets and edges */
    do_gfile('D',NULL);

  box_flag = old_box_flag;
} // end void detorus()

/**************************************************************************
* function: detorus_start()
* purpose: start function for do_gfile() generation of elements.
*/
void detorus_start()
{ int i;
  int ctrl_pts = web.skel[EDGE].ctrlpts;
  
  dt_valloc = 2*web.skel[VERTEX].count;
  dt_vlist = (struct dt_vertex*)temp_calloc(dt_valloc,sizeof(struct dt_vertex));
  dt_vcount = 0;
  dt_x_space = (REAL*)temp_calloc(dt_valloc,SDIM*sizeof(REAL));
  for ( i = 0 ; i < dt_valloc ; i++ )
    dt_vlist[i].x = dt_x_space + SDIM*i;

  dt_ealloc = 2*web.skel[EDGE].count;
  dt_elist = (struct dt_edge*)temp_calloc(dt_ealloc,sizeof(struct dt_edge));
  dt_ecount = 1;  /* start at 1 so can use signed indexes */
  dt_v_space = (int*)temp_calloc(dt_ealloc,ctrl_pts*sizeof(int));
  for ( i = 0 ; i < dt_ealloc ; i++ )
    dt_elist[i].v = dt_v_space + ctrl_pts*i;
    
  dt_falloc = 2*web.skel[FACET].count;
  dt_flist = (struct dt_facet*)temp_calloc(dt_falloc,sizeof(struct dt_facet));
  dt_fcount = 0;

} // end detorus_start()

/**************************************************************************
* function: detorus_vertex()
* purpose: vertex function for do_gfile() generation of elements, via
*     dt_edge().
*/
void detorus_vertex(
  struct graphdata *g,
  vertex_id v_id
)
{ int i;
  struct dt_vertex *vs;

  if ( dt_vcount >= dt_valloc )
  { dt_valloc *= 2;
    dt_vlist = (struct dt_vertex*)temp_realloc((char*)dt_vlist,
       dt_valloc*sizeof(struct dt_vertex));
    dt_x_space = (REAL*)temp_realloc((char*)dt_x_space,dt_valloc*SDIM*sizeof(REAL));
    for ( i = 0 ; i < dt_valloc ; i++ )
      dt_vlist[i].x = dt_x_space + SDIM*i;
  }

  vs = dt_vlist + dt_vcount;

  for ( i = 0 ; i < SDIM ; i++ )
    vs->x[i] = g->x[i];

  // See if same coordinates as proposed old v_id
  vs->v_id = NULLID;
  if ( valid_id(v_id) )
  { REAL *x;
    REAL diff;
    vs->orig_v_id = v_id;
    x = get_coord(v_id);
    for ( i = 0, diff = 0 ; i < SDIM ; i++ )
      diff += fabs(x[i] - g->x[i]);
    if ( diff < dt_eps )
      vs->v_id = v_id;
  }

  dt_vcount++;
} // end detorus_vertex()   
    
/**************************************************************************
* function: detorus_edge()
* purpose: edge function for do_gfile() generation of elements.
*/
void detorus_edge(
  struct graphdata *g,
  edge_id e_id
)
{
  struct dt_edge   *es;
   
  if ( dt_ecount >= dt_ealloc )
  { int ctrl_pts = web.skel[EDGE].ctrlpts;
    int i;
    
    dt_ealloc *= 2;
    dt_elist = (struct dt_edge*)temp_realloc((char*)dt_elist,
       dt_ealloc*sizeof(struct dt_edge));
    dt_v_space = (int*)temp_realloc((char*)dt_v_space,dt_ealloc*ctrl_pts*sizeof(int));
    for ( i = 0 ; i < dt_ealloc ; i++ )
      dt_elist[i].v = dt_v_space + ctrl_pts*i;
  }
  
  es  = dt_elist + dt_ecount;
  es->e_id = g->id;  //  e_id;
 
  // create vertices
  es->v[0] = dt_vcount;
  es->v[1] = dt_vcount+1;

  detorus_vertex(g,g[0].v_id);
  detorus_vertex(g+1,g[1].v_id);

  dt_ecount++;
    
} // end detorus_edge()

/**************************************************************************
* function: detorus_facet()
* purpose: facet function for do_gfile() generation of elements.
*/
void detorus_facet(
  struct graphdata *g,
  facet_id f_id
  )
{ int i;
  struct dt_facet *fs;
  facet_id fe = NULLID;

  if ( web.representation != SOAPFILM )
    return;

  g[3] = g[0]; /* convenient wrap */

  if ( dt_fcount >= dt_falloc )
  { dt_falloc *= 2;
    dt_flist = (struct dt_facet*)temp_realloc((char*)dt_flist,
       dt_falloc*sizeof(struct dt_facet));
  }

  fs  = dt_flist + dt_fcount;
  fs->f_id = f_id;
  if ( valid_id(f_id) )
  {
    fe = get_facet_fe(f_id);
    for ( i = 0 ; i < FACET_VERTS ; i++ )
    { // only a possible progenator, in case of clipping
      fs->e[i] = dt_ecount;
      detorus_edge(g+i, (g[i].etype==INVISIBLE_EDGE)?NULLID:get_fe_edge(fe));
      fe = get_next_edge(fe);
    }
  }
  else
    for ( i = 0 ; i < FACET_VERTS ; i++ )
    { 
      fs->e[i] = dt_ecount;
      detorus_edge(g+i, NULLID);
    }

  fs->flags = g[0].flags;  // for COLORS_SWAPPED

  dt_fcount++;
} // end detorus_facet()

/**************************************************************************
* function: detorus_end()
* purpose: ending function for do_gfile() generation of elements.
*/
void detorus_end()
{ int *translate;
  int i,j,keep;
  vertex_id v_id;
  edge_id e_id;
  facet_id f_id;
  facetedge_id fe_id;
  
#ifdef _DEBUGX
 printf("start of detorus_end\n");
  detorus_dump();
#endif

  // Using NEWELEMENT attribute to tell which elements needed.
  MFOR_ALL_VERTICES(v_id)
	  unset_attr(v_id,NEWELEMENT);
  MFOR_ALL_EDGES(e_id)
	  unset_attr(e_id,NEWELEMENT);
  MFOR_ALL_FACETS(f_id)
	  unset_attr(f_id,NEWELEMENT);
  MFOR_ALL_FACETEDGES(fe_id)
	  unset_attr(fe_id,NEWELEMENT);

  translate = detorus_unify_vertices();  

  /* convert vertex indices in edge list */
  for ( i = 1 ; i < dt_ecount ; i++ )
  { for ( j = 0 ; j < EDGE_VERTS ; j++ )
      dt_elist[i].v[j] = translate[dt_elist[i].v[j]];
  }

  temp_free((char*)translate);
  translate = detorus_unify_edges();
  
  // translate edge lists in dt_flist
  for ( i = 0 ; i < dt_fcount ; i++ )
  { for ( j = 0 ; j < FACET_EDGES ; j++ )
      dt_flist[i].e[j] = translate[dt_flist[i].e[j]];
  }
  
#ifdef _DEBUGX
printf("After edge list\n");
  detorus_dump();
#endif
  
  if ( web.representation == STRING )
  { detorus_string_cleanup();
  }
  else if ( web.representation == SOAPFILM )
  {
    // Sort facet list to get rid of duplicates with same edges
    qsort(dt_flist,dt_fcount,sizeof(struct dt_facet), FCAST dt_fcompare);
    for ( i = 1, keep = 0 ; i < dt_fcount ; i++ )
    { if ( dt_fcompare(dt_flist+i,dt_flist+keep) != 0 )
        dt_flist[++keep] = dt_flist[i];
    }
    dt_fcount = keep + 1;

    // Massage facet list
    for ( i = 0 ; i < dt_fcount ; i++ )
    { struct dt_facet *fs = dt_flist + i;

      // check for degenerate edges
      if ( !fs->e[0] || !fs->e[1] || !fs->e[2] )
      { fs->f_id = NULLID;
        continue;
      }

      detorus_handle_facet(fs);

    }
#ifdef _DEBUGX
   printf("After facet list \n");
    detorus_dump();
#endif

    detorus_soapfilm_cleanup();
  }
  
  torus_display_mode = TORUS_DEFAULT_MODE;
  web.torus_clip_flag = 0;
  web.torus_body_flag = 0;

  temp_free((char*)dt_vlist);
  temp_free((char*)dt_x_space);
  temp_free((char*)dt_elist);
  temp_free((char*)dt_v_space);
  temp_free((char*)dt_flist);
  temp_free((char*)translate);

}  // end detorus_end()

/*********************************************************************
*
* function: detorus_unify_vertices()
*
* purpose: sort vertex list, merge duplicates, and 
*          allocate new vertices if needed.
* return: translation list from original positions to merged.
*/
int * detorus_unify_vertices()
{ int i,j,k,keep;
  int *translate;
  int tcount;

  /* unify vertices */
  for ( i = 0 ; i < dt_vcount ; i++ )
    dt_vlist[i].orig = i;
  // sort along random direction
  qsort((char*)dt_vlist,dt_vcount,sizeof(struct dt_vertex),
      FCAST dt_vertex_comp);
  translate = (int*)temp_calloc(dt_vcount,sizeof(int));
  translate[dt_vlist[0].orig] = 0;
  for ( i = 1, keep = 0 ; i < dt_vcount ; i++ )
  { // hunt down keep list as long as within range
    for ( j = keep ; j >= 0 ; j-- )
    { int retval = dt_vertex_diff(dt_vlist+i,dt_vlist+j);
      if ( retval == DT_MERGEABLE )
      { // merge with previous
        // Take average of coordinates
        for ( k = 0 ; k < SDIM ; k++ )
          dt_vlist[j].x[k] = (dt_vlist[j].x[k]*dt_vlist[j].merge_count + dt_vlist[i].x[k])/
             (dt_vlist[j].merge_count + 1);
        translate[dt_vlist[i].orig] = j;
        dt_vlist[j].merge_count++;
        break;
      }
      else if ( retval == DT_OUT_OF_RANGE )
        j = -1; // force break 
    }
    if ( j < 0 )
    { // new keep list entry
      dt_vlist[++keep] = dt_vlist[i];
      dt_vlist[keep].merge_count = 1;
      translate[dt_vlist[i].orig] = keep;
    }
  }
  tcount = dt_vcount;
  dt_vcount = keep+1;
  
  // Sort keep vertices based on original v_id, so that
  // slight movements of the surface will have the 
  // same unwrapped structure.
  for ( i = 0 ; i < dt_vcount ; i++ )
    dt_vlist[i].orig = i;
  qsort(dt_vlist,dt_vcount,sizeof(struct dt_vertex),FCAST dt_vid_comp);
  for ( i = 0 ; i < dt_vcount ; i++ )
    dt_vlist[dt_vlist[i].orig].newspot = i;
  for ( i = 0 ; i < tcount ; i++ )
    translate[i] = dt_vlist[translate[i]].newspot;

  // Make new vertices as needed

  for ( i = 0 ; i < dt_vcount ; i++ )
  { REAL *z;
    struct dt_vertex *vs = dt_vlist + i;
   
    if ( valid_id(vs->v_id) )
    { /* see if same position as existing vertex */
      REAL diff = 0.0;
      REAL *x = get_coord(vs->v_id);

      for ( j = 0 ; j < SDIM ; j++ )
        diff += fabs(x[j] - vs->x[j]);
      if ( diff < dt_eps )
        continue;
    }

    if ( valid_id(vs->orig_v_id) )
    {
      vs->v_id = dup_vertex(vs->orig_v_id);

      // remove from any constraints or boundaries
      if ( get_vattr(vs->v_id) & BOUNDARY )
      { set_boundary_num(vs->v_id,0);
        unset_attr(vs->v_id,BOUNDARY|HIT_PARTNER);
      }
      else
        unset_v_all_constraints(vs->v_id);
 
      set_original(vs->v_id,vs->orig_v_id);  
      
      z = get_coord(vs->v_id);
      for ( j = 0 ; j < SDIM ; j++ )
        z[j] = vs->x[j];     
    }
    else 
      vs->v_id = new_vertex(vs->x,NULLID);
    
    set_attr(vs->v_id,NEWELEMENT); 
  }
  return translate;
} // end detorus_unify_vertices()

/***********************************************************************
*
* function: detorus_unify_edges()
*
* purpose: manipulate dt_elist to unify edges, creating new ones
*          as needed.
* 
* return: pointer to list of merged edges indexed by old edge numbers.
*/

int *detorus_unify_edges()
{ int i,j,keep;
  int ctrl_pts = web.skel[EDGE].ctrlpts;
  int headinx = (web.modeltype == LAGRANGE) ? ctrl_pts-1 : 1;
  int *translate;

  // Unify edges.  Note we still have edges in both directions.
  // Not using dt_elist[0] so can used signed edge indices in facets.
  for ( i = 1 ; i < dt_ecount ; i++ )
    dt_elist[i].orig = i;
  qsort((char*)(dt_elist+1),dt_ecount-1,sizeof(struct dt_edge),
      FCAST dt_edge_comp);

  translate = (int*)temp_calloc(dt_ecount,sizeof(int));
  translate[dt_elist[1].orig] = inverted(dt_elist[1].e_id) ? -1 : 1;
  for ( i = 1, keep = 0 ; i < dt_ecount ; i++ )
  { // check for degenerate edges and omit
    if ( equal_id(dt_vlist[dt_elist[i].v[0]].v_id,dt_vlist[dt_elist[i].v[headinx]].v_id) )
    { translate[dt_elist[i].orig] = 0; 
      continue;
    }
    
    if ( keep > 0 )
    { // see if we have a new edge
      if ( dt_edge_comp(dt_elist+i,dt_elist+keep) != 0 )
        dt_elist[++keep] = dt_elist[i];
    }
    else
      dt_elist[++keep] = dt_elist[i];

    /* Store in translate list; might be in reverse orientation */
    if ( dt_elist[i].v[0] == dt_elist[keep].v[0] )
      translate[dt_elist[i].orig] = keep;
    else
      translate[dt_elist[i].orig] = -keep;
    if ( inverted(dt_elist[keep].e_id) )
      translate[dt_elist[i].orig] *= -1;
  }
  dt_ecount = keep+1;

  // Get existing edges in positive orientation
  for ( i = 1 ; i < dt_ecount ; i++ )
    if ( inverted(dt_elist[i].e_id) )
    { if ( web.modeltype == LAGRANGE )
      { for ( j = 0 ; j < ctrl_pts/2 ; j++ )
        { int tmp = dt_elist[i].v[0];
          dt_elist[i].v[0] = dt_elist[i].v[ctrl_pts-1-j];
          dt_elist[i].v[ctrl_pts-1-j] = tmp;
        }
      }
      else
      { int tmp = dt_elist[i].v[0];
        dt_elist[i].v[0] = dt_elist[i].v[1];
        dt_elist[i].v[1] = tmp;
      }
      invert(dt_elist[i].e_id);
    }
  
  // Create new edges as needed, and reset heads of existing edges as needed.
  for ( i = 1 ; i < dt_ecount ; i++ )
  { struct dt_edge *es = dt_elist + i;
    edge_id newe_id;
    int headinx = (web.modeltype == LAGRANGE) ? web.skel[EDGE].ctrlpts-1 : 1;
    
    if ( valid_id(es->e_id) )
    { // first see if same as old edge
      
      if ( equal_id(get_edge_tailv(es->e_id),dt_vlist[es->v[0]].v_id) 
       && equal_id(get_edge_headv(es->e_id),dt_vlist[es->v[headinx]].v_id) )
    { // keep old edge
    /* not worth the trouble of trying to keep old edge in case one
       endpoint has changed.
    
        vertex_id old_headv = get_edge_headv(es->e_id);
        vertex_id new_headv = dt_vlist[es->v[1]].v_id;
        if ( !equal_id(old_headv,new_headv) )
        { remove_vertex_edge(old_headv,inverse_id(es->e_id));
          set_edge_headv(es->e_id,new_headv);
          set_vertex_edge(new_headv,inverse_id(es->e_id));
        }
      */
      }
      else
      { newe_id = dup_edge(es->e_id);
        switch ( web.modeltype )
        { case LINEAR:
            set_edge_tailv(newe_id,dt_vlist[es->v[0]].v_id);
            set_edge_headv(newe_id,dt_vlist[es->v[1]].v_id);
            break;
          case QUADRATIC:
            set_edge_tailv(newe_id,dt_vlist[es->v[0]].v_id);           
            set_edge_headv(newe_id,dt_vlist[es->v[1]].v_id);
            free_element(get_edge_midv(newe_id));
            set_edge_midv(newe_id,dt_vlist[es->v[2]].v_id);
            break;
          case LAGRANGE:
          { vertex_id *v = get_edge_vertices(newe_id);
            set_edge_tailv(newe_id,dt_vlist[es->v[0]].v_id);
            set_edge_headv(newe_id,dt_vlist[es->v[headinx]].v_id);
            for ( j = 1 ; j < ctrl_pts-1 ; j++ )
            {
              v[j] = dt_vlist[es->v[j]].v_id;
              set_attr(v[j],Q_MIDEDGE);
              set_vertex_edge(v[j],newe_id);
            }
            break;
          }  
        } 

        if ( get_eattr(newe_id) & BOUNDARY )
        { set_edge_boundary_num(newe_id,0);
          unset_attr(newe_id,BOUNDARY);
        }
        else
          unset_e_all_constraints(newe_id);

        if ( web.representation == SOAPFILM )
        {
		  set_edge_fe(newe_id,NULLID);
        }
        else if ( web.representation == STRING )
        { // copy facet links
          facetedge_id start_fe,fe,new_start_fe=NULLID,new_fe,prev_fe;
          facet_id f_id;

          fe = start_fe = get_edge_fe(es->e_id);
          prev_fe = NULLID;
          if ( valid_id(start_fe) )
          { do
            { f_id = get_fe_facet(fe);
              new_fe = new_facetedge(f_id,newe_id);
              if ( equal_id(fe,start_fe) )
              { new_start_fe = new_fe;
                set_next_facet(new_fe,new_fe);
                set_prev_facet(new_fe,new_fe);
              }
              else
              { set_next_facet(new_fe,new_start_fe);
                set_prev_facet(new_fe,prev_fe);
                set_prev_facet(new_start_fe,new_fe);
                set_next_facet(prev_fe,new_fe);
              }
              // Maybe splice in where tail of old was
              if ( equal_id(get_edge_tailv(newe_id),get_edge_tailv(es->e_id)) )
              { facetedge_id ffe = get_prev_edge(fe);
                set_next_edge(ffe,new_fe);
                set_prev_edge(new_fe,ffe);
                set_prev_edge(fe,NULLID);
              }
              // Maybe splice in where head of old was
              if ( equal_id(get_edge_headv(newe_id),get_edge_headv(es->e_id)) )
              { facetedge_id ffe = get_next_edge(fe);
                set_prev_edge(ffe,new_fe);
                set_next_edge(new_fe,ffe);
                set_next_edge(fe,NULLID);
              }
   
              if ( !valid_id(get_prev_edge(new_fe)) )
                  set_facet_fe(f_id,new_fe);

             
              prev_fe = new_fe;
              fe = get_next_facet(fe);
            } while ( !equal_id(fe,start_fe));
            set_edge_fe(newe_id,new_start_fe);
          }

        }

        es->e_id = newe_id;
      }
    }
    else
    {    
      /* new edge */
      newe_id = new_edge(dt_vlist[es->v[0]].v_id,
                               dt_vlist[es->v[headinx]].v_id,NULLID);
      if ( web.modeltype == QUADRATIC )
      { vertex_id v_id = dt_vlist[es->v[2]].v_id;
        set_edge_midv(newe_id,v_id);
      }
      else if ( web.modeltype == LAGRANGE )
      { vertex_id *v = get_edge_vertices(newe_id);
        for ( j = 1 ; j < ctrl_pts-1 ; j++ )
        { v[j] = dt_vlist[es->v[j]].v_id;
          set_attr(v[j],Q_MIDEDGE);
          set_vertex_edge(v[j],newe_id);
        }
      }
      es->e_id = newe_id;
    }
    set_attr(es->e_id,NEWELEMENT);
  }
  return translate;
} // end detorus_unify_edges()

/*********************************************************************************
*
* function: detorus_handle_facet()
*
* purpose: treat one facet, making new one if necessary.
*/

void detorus_handle_facet(struct dt_facet *fs)
{   int j;
    facetedge_id fe[FACET_EDGES];
    facet_id newf = NULLID;
    int old_facet_flag = 0;
    edge_id e_id;

    if ( valid_id(fs->f_id) )
    {   
      /* see if same as original facet */
      facetedge_id fe,base_fe = get_facet_fe(fs->f_id);

      fe = get_facet_fe(fs->f_id);
      for ( j = 0 ; j < FACET_EDGES ; j++ )
      { if ( fs->e[j] == 0)
          return;  // skip facet with edge collapsed
      }
      for ( j = 0 ; j < FACET_EDGES ; j++ )
      {        e_id = dt_elist[abs(fs->e[j])].e_id;
        if ( fs->e[j] < 0 )
           invert(e_id);
        if ( !equal_id(get_fe_edge(fe),e_id) )
          break;
        fe = get_next_edge(fe);
      }
      if ( j == FACET_EDGES )
      { /* same edges, so same facet */
       
        // Set up facetedges with possibly changed edges
	    facetedge_id fe_id = base_fe;
	    for ( j = 0 ; j < FACET_EDGES ; j++ )
		{ 
          edge_id want_e; // the edge that should be there 
          edge_id have_e; // edge already there
 		
		  have_e = get_fe_edge(fe_id);
		  if ( fs->e[j] > 0 )
		    want_e = dt_elist[fs->e[j]].e_id;
		  else 
			want_e = inverse_id(dt_elist[-fs->e[j]].e_id);
		
          if ( !equal_id(want_e,have_e) )
          { 
            // remove fe_id from old edge
            facetedge_id prev,next,ffe;
            prev = get_prev_facet(fe_id);
            next = get_next_facet(fe_id);
            if ( !equal_id(prev,fe_id) )
            { set_next_facet(prev,next);
              set_prev_facet(next,prev);
              set_edge_fe(have_e,next);
            }
            else
            {
              set_edge_fe(have_e,NULLID);          
            }
            
            // install on new edge
	        ffe = get_edge_fe(want_e);
	        if ( valid_id(ffe) )
	        { /* insert in edge facet loop */
	          set_next_facet(fe_id,get_next_facet(ffe));
	       	  set_prev_facet(fe_id,ffe);
		      set_prev_facet(get_next_facet(ffe),fe_id);
		      set_next_facet(ffe,fe_id);
	        }
	        else
	        {
	          set_next_facet(fe_id,fe_id);
		      set_prev_facet(fe_id,fe_id);
	        }
	        set_edge_fe(want_e,fe_id);
	        set_fe_edge(fe_id,want_e);
          }
		  
		  fe_id = get_next_edge(fe_id);
		}
        set_attr(fs->f_id,NEWELEMENT);
        newf = fs->f_id;
        old_facet_flag = 1;
      }
      else 
      { newf = dup_facet(fs->f_id);
	    set_facet_fe(newf,NULLID);
        set_attr(newf,NEWELEMENT);
        fs->f_id = newf;
        if ( get_fattr(fs->f_id) & BOUNDARY )
        { set_facet_boundary_num(fs->f_id,0);
          unset_attr(fs->f_id,BOUNDARY);
        }
        else
          unset_f_all_constraints(fs->f_id);

      }
    }
    else
    {
      newf = new_facet();
      set_attr(newf,NEWELEMENT);
      fs->f_id = newf;
    }

    /* install edges */
    if ( old_facet_flag )
    {
      fe[0] = get_facet_fe(fs->f_id);
      fe[1] = get_next_edge(fe[0]);
      fe[2] = get_next_edge(fe[1]);
    }
    for ( j = 0 ; j < FACET_EDGES ; j++ )
    { int inx;
	  facetedge_id ffe;

      inx = fs->e[j];
      if ( inx < 0 )
        e_id = inverse_id(dt_elist[-inx].e_id);
      else
        e_id = dt_elist[inx].e_id;
      if ( !old_facet_flag )
        fe[j] = new_facetedge(newf,e_id);
      else
      { if ( equal_id(e_id,get_fe_edge(fe[j])) )
          continue;  // everything fine
        else
        { 
          set_fe_edge(fe[j],e_id);
        }
      }
 	  ffe = get_edge_fe(e_id);
	  if ( valid_id(ffe) )
	  { /* insert in edge facet loop */
	    set_next_facet(fe[j],get_next_facet(ffe));
		set_prev_facet(fe[j],ffe);
		set_prev_facet(get_next_facet(ffe),fe[j]);
		set_next_facet(ffe,fe[j]);
	  }
	  else
	  {
	    set_next_facet(fe[j],fe[j]);
		set_prev_facet(fe[j],fe[j]);
	  }
	  set_edge_fe(e_id,fe[j]);
    }
    if ( !old_facet_flag )
    {
	  set_facet_fe(newf,fe[0]);
      for ( j = 0 ; j < FACET_EDGES ; j++ )
      { set_next_edge(fe[j],fe[(j<2)?(j+1):0]);
        set_prev_edge(fe[j],fe[j?(j-1):2]);
      }
    }

    if ( web.modeltype == LAGRANGE )
    { // set up facet vertex list
      vertex_id *v = get_facet_vertices(newf);
      for ( j = 0 ; j < web.skel[FACET].ctrlpts ; j++ )
      { if ( !equal_id(v[j],dt_vlist[fs->fv[j]].v_id) )
        {
          v[j] = dt_vlist[fs->fv[j]].v_id;
          if ( !valid_id(get_vertex_edge(v[j])) )
          { set_attr(v[j],Q_MIDFACET);
            set_vertex_facet(v[j],newf); 
          }
        }
      }
    }

    if ( fs->flags & COLORS_SWAPPED )
      reverse_orientation_facet(newf);

}  // end detorus_handle_facet()

/***********************************************************************
*
* function: detorus_string_cleanup()
*
* purpose: General cleanup after all elements generated.
*/

void detorus_string_cleanup()
{ int i;
  edge_id e_id;
  vertex_id v_id;
  body_id b_id;
  
 /* get rid of all old stuff not used */

  MFOR_ALL_EDGES(e_id)
	if ( !(get_eattr(e_id) & NEWELEMENT) )
    { if ( web.representation == STRING )
      { // cut fe facet links first so dissolve doesn't mess up facet fe
        facetedge_id fe,start_fe;
        fe = start_fe = get_edge_fe(e_id);
        do
        { set_fe_facet(fe,NULLID);
          fe = get_next_facet(fe);
        } while ( !equal_id(fe,start_fe) );
		dissolve_edge(e_id);
      }
    }
  MFOR_ALL_VERTICES(v_id)
	if ( !(get_vattr(v_id) & NEWELEMENT) )
		  dissolve_vertex(v_id);
  
 
  // Unset torus model
  if ( web.symmetry_flag )
  {
    web.torus_flag = 0;
    web.full_flag = 0;
    web.symmetry_flag = 0;
    sym_flags = 0;
    symmetry_name = NULL;
    sym_wrap = identity_wrap;
    sym_form_pullback = identity_form_pullback;
    sym_inverse = identity_inverse;
    sym_compose = identity_compose;
    transform_gen_count -= 3;
    memset(torus_period_expr,0,sizeof(torus_period_expr));
    memset(torus_display_period_expr,0,sizeof(torus_display_period_expr));
    view_transform_gens = NULL;
  }
  transform_gen_expr("");


  FOR_ALL_BODIES(b_id)
    set_body_volconst(b_id,0.0);
  for ( i = 0 ; i < 5 ; i++ )
  { string_rebody();   // seems to be needed again in some cases
    merge_bodies();
    suppress_erroutstring = 1;
    if ( facet_body_check() == 0 )
       break;
    suppress_erroutstring = 0;
  } 

  // fix up facet lists around vertices
  make_vfacet_lists();
  
  suppress_erroutstring = 0;
  if ( i == 5 )
    kb_error(1936,"detorus: still body errors after 5 rebodies.\n",WARNING);
  recalc();
  
  
  if ( torus_display_mode == TORUS_CONNECTED_MODE )
  { // Get rid of outer body, as identified by negative volume
    FOR_ALL_BODIES(b_id)
    { if ( get_body_volume(b_id) < 0 )
        dissolve_body(b_id);
    }
  }
  

} // end detorus_string_cleanup()

/***********************************************************************
*
* function: detorus_soapfilm_cleanup()
*
* purpose: General cleanup after all elements generated.
*/

void detorus_soapfilm_cleanup()
{ int i;
  edge_id e_id;
  vertex_id v_id;
  facet_id f_id;
  body_id b_id;
  
 /* get rid of all old stuff not used */
  MFOR_ALL_FACETS(f_id)
	  if ( !(get_fattr(f_id) & NEWELEMENT) )
      { set_facet_body(f_id,NULLID);
        set_facet_body(inverse_id(f_id),NULLID);
	    dissolve_facet(f_id);
      }
  MFOR_ALL_EDGES(e_id)
	  if ( !(get_eattr(e_id) & NEWELEMENT) )
		  dissolve_edge(e_id);
  MFOR_ALL_VERTICES(v_id)
	  if ( !(get_vattr(v_id) & NEWELEMENT) )
		  dissolve_vertex(v_id);

  /* merge facets with common edges */
  if ( detorus_sticky )
  { MFOR_ALL_EDGES(e_id)
    { facetedge_id start_fe,fe,ffe;
      start_fe = fe = get_edge_fe(e_id);
      if ( !valid_id(start_fe) ) continue;
      do
      { ffe = get_next_facet(fe);
        while ( !equal_id(ffe,start_fe) )
        { if ( equal_id(get_fe_edge(get_next_edge(fe)),get_fe_edge(get_next_edge(ffe))) )
          { merge_facet(get_fe_facet(fe),get_fe_facet(ffe));
            break;
          }
          ffe = get_next_facet(ffe);
        } 
        fe = get_next_facet(fe);
      } while (!equal_id(fe,start_fe));
    }
  }

   
  // Unset torus model
  if ( web.symmetry_flag )
  {
    web.torus_flag = 0;
    web.full_flag = 0;
    web.symmetry_flag = 0;
    sym_flags = 0;
    symmetry_name = NULL;
    sym_wrap = identity_wrap;
    sym_form_pullback = identity_form_pullback;
    sym_inverse = identity_inverse;
    sym_compose = identity_compose;
    transform_gen_count -= 3;
    memset(torus_period_expr,0,sizeof(torus_period_expr));
    memset(torus_display_period_expr,0,sizeof(torus_display_period_expr));
     view_transform_gens = NULL;
  }
  if ( transform_gen_count )
    transform_gen_expr("");
  else
    transforms_flag = 0;

  /* straighten out facet order around edges */
  if ( web.representation == SOAPFILM )
  { edge_id e_id;
    MFOR_ALL_EDGES(e_id)
      raw_fe_reorder(e_id);
  }
  
  // fix up facet lists around vertices
  make_vfacet_lists();

  FOR_ALL_BODIES(b_id)
    set_body_volconst(b_id,0.0);


  for ( i = 0 ; i < 5 ; i++ )
  { int rebnum = rebody();   // seems to be needed again in some cases
    int mergenum = merge_bodies();
    if ( rebnum == 0 && mergenum == 0 )
      break;
  } 

  suppress_erroutstring = 0;
  if ( i == 5 )
    kb_error(5480,"detorus: still body errors after 5 rebodies.\n",WARNING);
  recalc();

  if ( torus_display_mode == TORUS_CONNECTED_MODE )
  { // Get rid of outer body, as identified by negative volume
    FOR_ALL_BODIES(b_id)
    { if ( get_body_volume(b_id) < 0 )
        dissolve_body(b_id);
    }
  }
} // end detorus_soapfilm_cleanup()

/***************************************************************************
*
* function: detorus_soapfilm_generate()
*
* Purpose: Generate transformed facets for quadratic and lagrange soapfilm 
* models.
*
*/

void detorus_soapfilm_generate()
{ int *vstarts,*estarts;
  vertex_id v_id;
  edge_id e_id;
  facet_id f_id;
  int i,j;
  int *translate;
  struct dt_facet dtf;
  int ctrl_pts = web.skel[EDGE].ctrlpts;
  int *etranslate;

 // Using NEWELEMENT attribute to tell which elements needed.
  MFOR_ALL_VERTICES(v_id)
	  unset_attr(v_id,NEWELEMENT);
  MFOR_ALL_EDGES(e_id)
	  unset_attr(e_id,NEWELEMENT);
  MFOR_ALL_FACETS(f_id)
	  unset_attr(f_id,NEWELEMENT);
	  
  // Allocate lists
  dt_valloc = dt_transform_count*web.skel[VERTEX].count+5;
  dt_vlist = (struct dt_vertex*)temp_calloc(dt_valloc,sizeof(struct dt_vertex));
  dt_vcount = 0;
  dt_x_space = (REAL*)temp_calloc(dt_valloc,SDIM*sizeof(REAL));
  for ( i = 0 ; i < dt_valloc ; i++ )
    dt_vlist[i].x = dt_x_space + SDIM*i;

  dt_ealloc = dt_transform_count*web.skel[EDGE].count+5;
  dt_elist = (struct dt_edge*)temp_calloc(dt_ealloc,sizeof(struct dt_edge));
  dt_ecount = 1;  /* start at 1 so can use signed indexes */
  dt_v_space = (int*)temp_calloc(dt_ealloc,web.skel[EDGE].ctrlpts*sizeof(int));
  for ( i = 0 ; i < dt_ealloc ; i++ )
    dt_elist[i].v = dt_v_space + ctrl_pts*i;

  vstarts = (int*)temp_calloc(web.skel[VERTEX].max_ord+5,sizeof(int));
  estarts = (int*)temp_calloc(web.skel[EDGE].max_ord+5,sizeof(int));
  
  // Generate all transforms
  FOR_ALL_VERTICES(v_id)
  { int ord = loc_ordinal(v_id);
    REAL *x = get_coord(v_id);
    vstarts[ord] = dt_vcount;
    x[SDIM] = 1.0;
    for ( i = 0 ;  i < dt_transform_count ; i++ )
    { dt_vlist[dt_vcount].v_id = v_id;
      dt_vlist[dt_vcount].orig_v_id = v_id;
      matvec_mul(view_transforms[i],x,dt_vlist[dt_vcount].x,
                  SDIM+1,SDIM+1);
      dt_vcount++;
    } 
  }
  translate = detorus_unify_vertices();
  
  FOR_ALL_EDGES(e_id)
  { int ord = loc_ordinal(e_id);
    vertex_id *v = get_edge_vertices(e_id);
    int ctrl_pts = web.skel[EDGE].ctrlpts;
    int maxi;

    estarts[ord] = dt_ecount;
    if ( get_eattr(e_id) & NO_TRANSFORM )
    { // see if any adjacent facets need it
      facetedge_id start_fe,fe_id;
      maxi = 1;
      start_fe = fe_id = get_edge_fe(e_id);
      if ( valid_id(start_fe) )
      do 
      { if ( get_eattr(get_fe_facet(fe_id)) & NO_TRANSFORM )
        {  maxi = dt_transform_count;
           break;
        }
        fe_id = get_next_facet(fe_id);
      } while ( !equal_id(fe_id,start_fe) );
    }
    else maxi = dt_transform_count;
    for ( i = 0 ;  i < maxi ; i++ )
    { dt_elist[dt_ecount].e_id = e_id;
      for ( j = 0 ; j < ctrl_pts ; j++ )
        dt_elist[dt_ecount].v[j] = translate[vstarts[loc_ordinal(v[j])] + i];
      dt_ecount++;
    } 
  }
  
  etranslate = detorus_unify_edges();
 
  FOR_ALL_FACETS(f_id)
  { int maxi;
    if ( get_fattr(f_id) & NEWELEMENT )
      continue;
    maxi = get_fattr(f_id) & NO_TRANSFORM ? 1 : dt_transform_count;
    for ( i = 0 ; i < maxi ; i++ )
    {
      facetedge_id fe;
      
      dtf.flags = 0;
      fe = get_facet_fe(f_id);
      e_id = get_fe_edge(fe);
      for ( j = 0 ; j < FACET_EDGES ; j++ )
      { dtf.e[j] = etranslate[estarts[loc_ordinal(e_id)] + i];
        if ( inverted(e_id) )
           dtf.e[j] *= -1;
        fe = get_next_edge(fe);
        e_id = get_fe_edge(fe);
      }
      if ( view_transform_det && ((view_transform_det[i] < 0)
         != (transform_colors[i] == SWAP_COLORS)) )
      { // inverted
        int tmp = dtf.e[0];
        dtf.e[0] = -dtf.e[2];
        dtf.e[1] = -dtf.e[1];
        dtf.e[2] = -tmp;
      }
      dtf.f_id = f_id;
      if ( web.modeltype == LAGRANGE )
      { int fv_space[MAXLAGRANGE*MAXLAGRANGE];
        vertex_id *v = get_facet_vertices(f_id);
        dtf.fv = fv_space;
        for ( j = 0 ; j < web.skel[FACET].ctrlpts ; j++ )
          dtf.fv[j] = translate[vstarts[loc_ordinal(v[j])] + i];
        if ( view_transform_det && view_transform_det[i] < 0 )
        { // inverted
          int ii;
          for ( j = 0 ; j < web.lagrange_order/2 ; j++ )
            for ( ii = j+1 ; ii+j <= web.lagrange_order ; ii++ )
            { int k = ii + j*(web.lagrange_order+1) - (j*(j-1))/2;
              int kk = j + ii*(web.lagrange_order+1) - (ii*(ii-1))/2;
              int tmp = dtf.fv[k];
              dtf.fv[k] = dtf.fv[kk];
              dtf.fv[kk] = tmp;
            }

        }
      }
      detorus_handle_facet(&dtf);   

    }
  }
  
  detorus_soapfilm_cleanup();
  
  temp_free((char*)dt_vlist);
  temp_free((char*)dt_x_space);
  temp_free((char*)dt_elist);
  temp_free((char*)dt_v_space);
  temp_free((char*)translate);
  temp_free((char*)etranslate);

 } // end detorus_soapfilm_generate()
  

 // detorus string version
 /***************************************************************************
*
* function: detorus_string_generate()
*
* Purpose: Generate transformed facets for quadratic and lagrange soapfilm 
* models.
*
*/

void detorus_string_generate()
{ int *vstarts,*estarts;
  vertex_id v_id;
  edge_id e_id;
  int i,j;
  int *translate;
  int ctrl_pts = web.skel[EDGE].ctrlpts;
  int *etranslate;

 // Using NEWELEMENT attribute to tell which elements needed.
  MFOR_ALL_VERTICES(v_id)
	  unset_attr(v_id,NEWELEMENT);
  MFOR_ALL_EDGES(e_id)
	  unset_attr(e_id,NEWELEMENT);
	  
  // Allocate lists
  dt_valloc = dt_transform_count*web.skel[VERTEX].count+5;
  dt_vlist = (struct dt_vertex*)temp_calloc(dt_valloc,sizeof(struct dt_vertex));
  dt_vcount = 0;
  dt_x_space = (REAL*)temp_calloc(dt_valloc,SDIM*sizeof(REAL));
  for ( i = 0 ; i < dt_valloc ; i++ )
    dt_vlist[i].x = dt_x_space + SDIM*i;

  dt_ealloc = dt_transform_count*web.skel[EDGE].count+5;
  dt_elist = (struct dt_edge*)temp_calloc(dt_ealloc,sizeof(struct dt_edge));
  dt_ecount = 1;  /* start at 1 so can use signed indexes */
  dt_v_space = (int*)temp_calloc(dt_ealloc,web.skel[EDGE].ctrlpts*sizeof(int));
  for ( i = 0 ; i < dt_ealloc ; i++ )
    dt_elist[i].v = dt_v_space + ctrl_pts*i;

  vstarts = (int*)temp_calloc(web.skel[VERTEX].max_ord+5,sizeof(int));
  estarts = (int*)temp_calloc(web.skel[EDGE].max_ord+5,sizeof(int));
  
  // Generate all transforms
  FOR_ALL_VERTICES(v_id)
  { int ord = loc_ordinal(v_id);
    REAL *x = get_coord(v_id);
    vstarts[ord] = dt_vcount;
    x[SDIM] = 1.0;
    for ( i = 0 ;  i < dt_transform_count ; i++ )
    { dt_vlist[dt_vcount].v_id = v_id;
      dt_vlist[dt_vcount].orig_v_id = v_id;
      matvec_mul(view_transforms[i],x,dt_vlist[dt_vcount].x,
                  SDIM+1,SDIM+1);
      dt_vcount++;
    } 
  }
  translate = detorus_unify_vertices();
  
  FOR_ALL_EDGES(e_id)
  { int ord = loc_ordinal(e_id);
    vertex_id *v = get_edge_vertices(e_id);
    int ctrl_pts = web.skel[EDGE].ctrlpts;
    int maxi;

    estarts[ord] = dt_ecount;

    maxi = get_fattr(e_id) & NO_TRANSFORM ? 1 : dt_transform_count;
    for ( i = 0 ; i < maxi ; i++ )
    { dt_elist[dt_ecount].e_id = e_id;
      for ( j = 0 ; j < ctrl_pts ; j++ )
        dt_elist[dt_ecount].v[j] = translate[vstarts[loc_ordinal(v[j])] + i];
      dt_ecount++;
    } 
  }
  
  etranslate = detorus_unify_edges();
 
  
  detorus_string_cleanup();

  temp_free((char*)dt_vlist);
  temp_free((char*)dt_x_space);
  temp_free((char*)dt_elist);
  temp_free((char*)dt_v_space);
  temp_free((char*)translate);
  temp_free((char*)etranslate);

 } // end detorus_string_generate()
  
