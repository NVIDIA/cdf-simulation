/*************************************************************
*  This file is part of the Surface Evolver source code.     *
*  Programmer:  Ken Brakke, brakke@susqu.edu                 *
*************************************************************/


/*****************************************************************
*
*  File: painter.c 
*
*  Contents:  Routines to accumulate and sort facets for display
*                 back to front.  Not device specific, but calls
*                 device specific routines  for actual display.
*                 Also does transformations on edges.
*                 painter_end() uses Newell-Newell-Sancha algorithm
*                 to depth sort and display facets.
*                 This version uses a separate quadtree of depth-ordered 
*                 lists from the drawing-order list to minimize comparisons
*                 needed for large elements.
*/

/* Some timings with version 2.18j on starfish31.dmp  (2560 facets)

   Without visibility test:
   Shape     Fund.   time,sec    filesize
   Fund        1        0.093      190,829
   Cubelet    12        1.08     2,276,025
   Cube       48        4.64     9,100,293
   Rhomb      96        9.625   18,199,633

   With visibility test:
   Shape     Fund.   time,sec    filesize
   Fund        1        0.165      168,332
   Cubelet    12        1.89     1,444,778
   Cube       48        9.56     4,456,944
   Rhomb      96       24.7      7,162,980
*/

#include "include.h"

static size_t count;      /* number of facets */
static size_t maxcount;  /* number allocated */
struct tsort *trilist; /* list for depth sorting triangles */
static int gdim = 3;      /* dimension doing graphics in */
static int painter_multiple_sweep_flag;  // for really big surfaces
#define MULTIPLE_SWEEP_PREP 2
#define MULTIPLE_SWEEP_GO   3
#define MULTIPLE_SWEEP_DONE 4
#define SWEEPBINS 1000
static size_t sweep_mins[SWEEPBINS], sweep_maxs[SWEEPBINS];
static REAL sweep_low = -10.0;
static REAL sweep_high = 10.0;
static REAL sweep_delta;
static REAL sweep_top; //  current cutoff min
static REAL sweep_bottom; // previous cutoff
static size_t sweep_k;  // keeping track in sweep_mins[]
static size_t sweep_total,sweep_done; // for reporting progress

/* results of comparing depth of facets */
#define  DISJOINT      1
#define  FIRST_BACK    2
#define  SECOND_BACK   4
#define  ASPLITTINGB   8
#define  BSPLITTINGA  16
#define  NOTKNOWN     32
#define  COPLANAR     64

#define TEXTRA 100
static  struct tsort textra[TEXTRA]; /* for triangle fragments */

/* for debugging; just displays list as is after given number of facets */
size_t debug_k = 0x7FFFFFFF;

REAL tableau[7][3];  /* simplex tableau */
void pivot(int ,int );
int newell_split(struct tsort *,struct tsort *,struct tsort *,struct tsort *);
int backstamp;   /* for timestamping being at back of list */

int plane_test(struct tsort *,struct tsort *);
int setquadcode(struct tsort *);
void find_bbox(struct tsort *);
int separating_line(struct tsort*,struct tsort*);
int separating_plane(struct tsort*,struct tsort*,int);

static int ttcompare(  /* depth comparison for sort */
  struct tsort **t1,
  struct tsort **t2
)
{
  /* First, compare back z */
  if ( t1[0]->mins[2] > t2[0]->mins[2] ) return 1;
  if ( t1[0]->mins[2] < t2[0]->mins[2] ) return -1;
  /* Break ties with front z */
  if ( t1[0]->maxs[2] > t2[0]->maxs[2] ) return 1;
  if ( t1[0]->maxs[2] < t2[0]->maxs[2] ) return -1;
  /* Finally, break ties with id to get consistent ordering,
     independent of how many elements are being sorted */
  if ( t1[0]->f_id > t2[0]->f_id ) return 1;
  if ( t1[0]->f_id < t2[0]->f_id ) return -1;
  return 0;

} // end ttcompare()

/* quadtree of depth lists */
struct qtree_t { struct tsort *depthhead;
                 float maxdepth; /* of all in subtree */
} *qtree;
int maxquaddepth = 8; /* maximum depth of quadtree */
int get_quadindex(unsigned int);
void qdepth_insert(struct tsort *);
struct tsort *search_subtree(int,struct tsort *,int *);

/* visibility stuff */
void visibility_stage(struct tsort *);
void visibility_end(void);
size_t vis_count;  /* used structures */
size_t vis_max;   /* allocated structures */
struct tsort *vis_list;  /* storage */


/*****************************************************************
*
*  Function: find_bbox()
*
*  Purpose: find 3D bounding box for edge or facet.
*
*/

void find_bbox(struct tsort *t)
{ int n;
  REAL dx,dy,len;

  if ( (t->flag & 0xF) == EDGE )
  {
    if ( t->flag & EDGE_ARC )
    { REAL w1[MAXCOORD],w2[MAXCOORD],mag1,mag2,w1w2,center[2],radius;
      REAL det,angle1,angle2;
      int i;
      for (i = 0 ; i < SDIM ; i++ )
      { w1[i] = t->x[1][i] - t->x[0][i];
        w2[i] = t->x[2][i] - t->x[0][i];
      }
      det = w1[0]*w2[1] - w1[1]*w2[0];
      mag1 = SDIM_dot(w1,w1); mag2 = SDIM_dot(w2,w2);
      w1w2 = w1[0]*w2[0] + w1[1]*w2[1];
      for ( n = 0 ; n < gdim ; n++ )
      { /* endpoints first */
        if ( t->x[0][n] < t->x[2][n] )
         { t->maxs[n] = t->x[2][n];
           t->mins[n] = t->x[0][n];
         }
         else
         { t->maxs[n] = t->x[0][n];
           t->mins[n] = t->x[2][n];
         }
       }
      if ( 4000*det*det > mag1*mag1*mag2 + mag1*mag2*mag2 - 2*mag1*w1w2*mag2 )
      { /* circle rather that straight line */
        center[0] = t->x[0][0] + 0.5*(w2[1]*mag1 - w1[1]*mag2)/det;
        center[1] = t->x[0][1] + 0.5*(-w2[0]*mag1 + w1[0]*mag2)/det;
        radius =  sqrt((mag1*mag1*mag2+mag1*mag2*mag2-2*mag1*w1w2*mag2)
                         /4/det/det);
        angle1 = atan2(t->x[0][1]-center[1],t->x[0][0]-center[0]);
        angle2 = atan2(t->x[2][1]-center[1],t->x[2][0]-center[0]);
        if ( det < 0 )
        { REAL temp = angle1; angle1 = angle2; angle2 = temp; }
        if ( angle2 < angle1 ) angle2 += 2*M_PI;
        if ( (angle1 < 0.0 && angle2 > 0.0 ) || 
             (angle1 < 2*M_PI && angle2 > 2*M_PI) )
          t->maxs[0] = (float)(center[0] + radius);
        if ( (angle1 < M_PI && angle2 > M_PI ) || 
             (angle1 < 3*M_PI && angle2 > 3*M_PI) )
          t->mins[0] = (float)(center[0] - radius);
        if ( (angle1 < M_PI/2 && angle2 > M_PI/2 ) || 
             (angle1 < 5*M_PI/2 && angle2 > 5*M_PI/2) )
          t->maxs[1] = (float)(center[1] + radius);
        if ( (angle1 < -M_PI/2 && angle2 > -M_PI/2 ) || 
             (angle1 < 3*M_PI/2 && angle2 > 3*M_PI/2) )
          t->mins[1] = (float)(center[1] - radius);
        return;
      } 
    } /* end EDGE_ARC */
    else /* just a straight segment */
    for ( n = 0 ; n < gdim ; n++ )
    { if ( t->x[0][n] < t->x[1][n] )
      { t->maxs[n] = t->x[1][n];
        t->mins[n] = t->x[0][n];
      }
      else
      { t->maxs[n] = t->x[0][n];
        t->mins[n] = t->x[1][n];
      }
    }
    /* adjust extents for thickness */
    dx = fabs(t->x[1][0] - t->x[0][0]);
    dy = fabs(t->x[1][1] - t->x[0][1]);
    len = sqrt(dx*dx+dy*dy);
    if ( len > 0.0 )
    { t->maxs[0] += (float)(dy/len*t->width/2);
      t->mins[0] -= (float)(dy/len*t->width/2);
      t->maxs[1] += (float)(dx/len*t->width/2);
      t->mins[1] -= (float)(dx/len*t->width/2);
    }
    return;
  } /* end EDGE */

  /* facet */
  for ( n = 0 ; n < gdim ; n++ )
  { if ( t->x[0][n] < t->x[1][n] )
    { if  ( t->x[2][n] < t->x[0][n] )
      { t->maxs[n] = t->x[1][n];
        t->mins[n] = t->x[2][n];
      }
      else if ( t->x[1][n] < t->x[2][n] )
      { t->maxs[n] = t->x[2][n];
        t->mins[n] = t->x[0][n];
      }
      else
      { t->maxs[n] = t->x[1][n];
        t->mins[n] = t->x[0][n];
      }
    }
    else
    { if  ( t->x[2][n] < t->x[1][n] )
      { t->maxs[n] = t->x[0][n];
        t->mins[n] = t->x[2][n];
      }
      else if ( t->x[0][n] < t->x[2][n] )
      { t->maxs[n] = t->x[2][n];
        t->mins[n] = t->x[1][n];
      }
      else
      { t->maxs[n] = t->x[0][n];
        t->mins[n] = t->x[1][n];
      }
    }
  }
} // end find_bbox()

/*************************************************************************
* Function: setquadcode()
* Purpose:  For setting quadtree code and checking if in bounding box. 
* Coded for 32 bit ints. 
*/
#define OUTOFBOX 0
#define INTHEBOX 1

int setquadcode(struct tsort *t)
{ unsigned int q = 0;  /* the quadcode */
  unsigned int bit = 1;  /* for shifting to quad bit position */
  int n;
  REAL midx = (minclipx+maxclipx)/2;
  REAL midy = (minclipy+maxclipy)/2;
  REAL deltax = (maxclipx-minclipx)/4;
  REAL deltay = (maxclipy-minclipy)/4;

  if ( t->maxs[0] < minclipx || t->mins[0] > maxclipx || t->maxs[1] < minclipy
       || t->mins[1] > maxclipy ) return OUTOFBOX;

  for ( n = 0  ; n < 8 ; n++, deltax /= 2, deltay /= 2 )
  {
    if ( t->maxs[0] <= midx ) 
    { q |= bit; midx -= deltax; }
    else if ( t->mins[0] >= midx )
    { q |= bit<<1; midx += deltax; }
    else break;
    bit <<= 2;
    if ( t->maxs[1] <= midy ) 
    { q |= bit; midy -= deltay; }
    else if ( t->mins[1] >= midy )
    { q |= bit<<1; midy += deltay; }
    else break;
    bit <<= 2;
  }

  t->quadcode = q;
  return INTHEBOX;
} // end setquadcode()

/*************************************************************************
* Function: multiple_sweep_prep()
*
* Purpose: Generate graphics items once to count and find bins.
*/
void multiple_sweep_prep()
{ size_t total;
  int k;

  bbox_minx = bbox_miny = 1e20;
  bbox_maxx = bbox_maxy = -1e20;

  memset((void*)sweep_mins,0,sizeof(sweep_mins));
  memset((void*)sweep_maxs,0,sizeof(sweep_maxs));
  sweep_delta = (sweep_high - sweep_low)/SWEEPBINS;
  painter_multiple_sweep_flag = MULTIPLE_SWEEP_PREP;
  raw_generate();

  // see if we are ever going to have more than maxcount
  // items in a sweep
  for ( total = 0, k = 0 ; k < SWEEPBINS ; k++ )
  { total += sweep_mins[k] - (k ? sweep_maxs[k-1] : 0);
    if ( total > maxcount )
      kb_error(5532,"depth sort: too many items in sweep.\n",
         RECOVERABLE);
    sweep_total += sweep_mins[k];
  }

  // Now set up for first sweep
  count = 0;
  sweep_done = 0;
  for ( total = 0, k = 0 ; k < SWEEPBINS ; k++ )
  { if ( total + sweep_mins[k] > maxcount )
       break;
    total += sweep_mins[k];
  }
  sweep_k = k;
  sweep_done = total;
  sweep_top = sweep_low + (k-1)*sweep_delta;
  sweep_bottom = -1e30;

  painter_multiple_sweep_flag = MULTIPLE_SWEEP_GO;
  (*init_graphics)();  // print out postscript header
  erroutstring("Starting first sweep.\n");

} // end multiple_sweep_prep()

/*************************************************************************
* Function: sweep_prep_one()
*
* Purpose: tally one tsort structure in initial sweep
*/
void sweep_prep_one(struct tsort *t)
{
    int bin = (int)((t->mins[SDIM-1] - sweep_low)/sweep_delta);
    if ( bin < 0 ) 
      bin = 0;
    if ( bin >= SWEEPBINS )
      bin = SWEEPBINS - 1;
    sweep_mins[bin]++;
    bin = (int)((t->maxs[SDIM-1] - sweep_low)/sweep_delta);
    if ( bin < 0 ) 
      bin = 0;
    if ( bin >= SWEEPBINS )
      bin = SWEEPBINS - 1;
    sweep_maxs[bin]++;

  /* find bounding box */
  if ( need_bounding_box )
  { 
    { if ( t->mins[0] < bbox_minx ) bbox_minx = (REAL)t->mins[0];
      if ( t->mins[1] < bbox_miny ) bbox_miny = (REAL)t->mins[1];
      if ( t->maxs[0] > bbox_maxx ) bbox_maxx = (REAL)t->maxs[0];
      if ( t->maxs[1] > bbox_maxy ) bbox_maxy = (REAL)t->maxs[1];
    }
  }

} // end sweep_prep_one()

/*************************************************************************
* Function: reset_multiple_sweep()
*
* Purpose: reset trilist in preparation for next sweep of multiple sweep
*/
void reset_multiple_sweep()
{ size_t k,j,total;

  sprintf(msg,"Sweep completed. Done %lld of %lld items.\n",(long long)sweep_done,
      (long long)sweep_total);
  erroutstring(msg);

  if ( sweep_k >= SWEEPBINS )
  { painter_multiple_sweep_flag = MULTIPLE_SWEEP_DONE;
    return;
  }

  // compactify left-over trilist, with extra ones tacked on at end
  for ( k = 0, j = 0 ; k < count ; k++ )
    if ( trilist[k].flag )
      trilist[j++] = trilist[k];
  for ( k = 0 ; k < TEXTRA ; k++ )
    if ( textra[k].flag )
      trilist[j++] = textra[k];
  memset((void*)(trilist+j),0,(maxcount-j)*sizeof(struct tsort));
  memset((void*)(textra),0,TEXTRA*sizeof(struct tsort));
  count = j-TEXTRA;

  // Now get bounds for next sweep
  for ( total = count, k = sweep_k ; k < SWEEPBINS ; k++ )
  { if ( total + sweep_mins[k] > maxcount )
       break;
    total += sweep_mins[k];
    sweep_done += sweep_mins[k];
  }
  sweep_k = k;
  sweep_bottom = sweep_top;
  if ( sweep_k >= SWEEPBINS )
    sweep_top = 1e30;
  else 
    sweep_top = sweep_low + (k-1)*sweep_delta;

} // end reset_multiple_sweep()

/*************************************************************************
* Function: painter_start()
* Purpose:  Start routine called by graphgen()
*/
void painter_start()
{ int dummy;
  size_t allocsize;

  gdim = (SDIM <= 3) ? SDIM : 3;

  /* allocate space for depth sort list */
  if ( web.representation == STRING )
  {  maxcount = web.skel[EDGE].count + 5;
     if ( web.torus_flag ) maxcount *= 2;
  }
  else
  { if ( web.torus_flag )
      if ( torus_display_mode == TORUS_CLIPPED_MODE )
        maxcount = 5*web.skel[FACET].count+ bare_edge_count + 5;
      else maxcount = 2*web.skel[FACET].count+ bare_edge_count + 5;
    else
      maxcount = web.skel[FACET].count + bare_edge_count + 5;
  }
  if ( transforms_flag ) maxcount *= transform_count;
  if ( web.dimension > 2 )
     maxcount *= web.dimension+1; /* each simplex face becomes facet */
  allocsize = (long)maxcount*sizeof(struct tsort);
  painter_multiple_sweep_flag = 0;
#ifdef WIN32
  { MEMORYSTATUSEX statex;
    statex.dwLength = sizeof (statex);
    GlobalMemoryStatusEx (&statex);
    if ( allocsize > statex.ullAvailPhys/4 )
    { painter_multiple_sweep_flag = 1;
      maxcount = (size_t)(statex.ullAvailPhys/4/sizeof(struct tsort));
    }
  }
#elif defined(LINUX) && !defined(MAC_OS_X)
  { struct sysinfo s;
    sysinfo(&s);
    if ( allocsize > (size_t)(s.freeram)*s.mem_unit/4 )
    { painter_multiple_sweep_flag = 1;
      maxcount = (size_t)(s.freeram)*s.mem_unit/4;
    }
  }
#else
  if ( maxcount >= 1000000  )
  { painter_multiple_sweep_flag = 1;
    maxcount = 1000000;
  }
#endif
  trilist = (struct tsort *)temp_calloc(maxcount,sizeof(struct tsort));
  count = 0;
  if ( painter_multiple_sweep_flag )
  { erroutstring("Very big surface, so painter algorithm doing multiple sweeps.\n");
    multiple_sweep_prep();
  }
  vis_count = 0;
  vis_list = NULL;
  vis_max = 0;
  backstamp = 0;
  ps_widthattr = find_extra(PS_WIDTHNAME,&dummy);
} // end painter_start()

/****************************************************************************
*
* Function: painter_edge()
*
* Purpose: Per-edge function called by graphgen()
*/
void painter_edge(
  struct graphdata *gdata,
  edge_id e_id
)
{
  struct tsort *t;
  int i,j;
  REAL a[MAXCOORD+1],b[MAXCOORD+1];
  REAL dx,dy,dz,mag,width;
  int ctrl_pts = gdata[0].flags & EDGE_ARC ? 3 : 2;

  if ( gdata->color == CLEAR ) return;
  if ( count >= maxcount-2 )
  { trilist = (struct tsort *)temp_realloc((char*)trilist,
         (maxcount+200)*sizeof(struct tsort));
    maxcount += 200;
  }

  t = trilist + count;
  t->flag = EDGE;
  t->flag |= (gdata->flags & (EDGE_ARC|LABEL_EDGE|LABEL_HEAD|LABEL_TAIL));
  for ( j =  SDIM ; j < HOMDIM-1 ; j++ ) a[j] = 0.0; /* filler */
  for ( i = 0 ; i < ctrl_pts ; i++ )
  {
    for ( j = 0 ; (j < SDIM) && (j < HOMDIM-1) ; j++ ) 
        a[j] = gdata[i].x[j];
    a[HOMDIM-1] = 1.0;
    matvec_mul(view,a,b,HOMDIM,HOMDIM);  /* transform */
    if ( SDIM <= 2 )
      for ( j = 0 ; j < 3 ; j++ ) t->x[i][j] = (float)b[j];
    else  for ( j = 0 ; j < 3 ; j++ ) t->x[i][j] = (float)b[(j+1)%3];
    t->x[i][2] += (float).0001;   /* bias edges in front of facets */
  }

  /* width of edge, in descending order of thickness */
  if ( ps_widthattr >= 0 )
    width = *EREAL(t->f_id,ps_widthattr);
  else if ( gdata->etype & BARE_EDGE ) width = ps_bareedgewidth;
  else if ( gdata->etype & FIXED_EDGE ) width = ps_fixededgewidth;
  else if ( gdata->etype & CONSTRAINT_EDGE ) width = ps_conedgewidth;
  else if ( gdata->etype & BOUNDARY_EDGE ) width = ps_conedgewidth;
  else if ( gdata->etype & SINGLE_EDGE ) width = ps_stringwidth;
  else if ( gdata->etype & TRIPLE_EDGE ) width = ps_tripleedgewidth;
  else width = ps_gridedgewidth; /* regular grid interior edge */
  t->width = (float)width;

  t->f_id = e_id;
  t->color = t->ecolor[0] =  gdata->ecolor;
  t->etype[0] = gdata->etype;
  /* find extents */
  find_bbox(t);

  if ( painter_multiple_sweep_flag == MULTIPLE_SWEEP_PREP )
  { sweep_prep_one(t);
    return;
  }
  else if ( painter_multiple_sweep_flag == MULTIPLE_SWEEP_GO )
  { if ( t->mins[SDIM-1] > sweep_top || t->mins[SDIM-1] <= sweep_bottom ) 
      return;
  }

  /* normal vector, closest to z axis */
  dx = t->x[1][0] - t->x[0][0];
  dy = t->x[1][1] - t->x[0][1];
  dz = t->x[1][2] - t->x[0][2];
  t->normal[0] = (float)(dx*dz);
  t->normal[1] = (float)(dy*dz);
  t->normal[2] = -(float)(dx*dx+dy*dy);
  mag = sqrt(dotf(t->normal,t->normal,3));
  if ( mag != 0.0 )
      for ( i = 0 ; i < 3; i++ ) t->normal[i] /= (float)mag;

  if ( setquadcode(t) == OUTOFBOX ) return;
  count++;
} // end painter_edge()

/********************************************************************
*
* Function: painter_facet()
*
* Purpose: Per-facet function called by graphgen()
*/
void painter_facet(
  struct graphdata *gdata,
  facet_id f_id
)
{
  int i,j;
  REAL a[MAXCOORD+1],b[FACET_VERTS][MAXCOORD+1];
  struct tsort *t;
  REAL normal[MAXCOORD],mag;

  if ( gdata[0].color == UNSHOWN )
  { /* just do edges */
    struct graphdata ggdata[2];
    facetedge_id fe=NULLID;
    if ( valid_id(f_id) )
      fe = get_facet_fe(f_id);
    for ( i = 0 ; i < FACET_EDGES ; i++ )
    { if ( (gdata[i].etype&EBITS) == INVISIBLE_EDGE ) continue;
      ggdata[0] = gdata[i]; 
      ggdata[0].color = gdata[i].ecolor;
      ggdata[1] = gdata[i==2 ? 0 : i+1];
      if ( valid_id(fe) )
      { painter_edge(ggdata,get_fe_edge(fe));
        fe = get_next_edge(fe);
      }
      else painter_edge(ggdata,NULLID);
    }
    return;
  }

  if ( count >= maxcount )
  { trilist = (struct tsort *)temp_realloc((char*)trilist,
         (maxcount+200)*sizeof(struct tsort));
    maxcount += 200;
  }
  t = trilist + count;
  t->flag = FACET | (gdata->flags&LABEL_FACET);
  t->f_id = f_id;
  t->color = gdata[0].backcolor;  /* not sure why, but works */
  for ( i = 0 ; i < FACET_EDGES ; i++ ) 
  { t->ecolor[i] = gdata[i].ecolor;
    t->etype[i] = gdata[i].etype;
    t->v_id[i] = gdata[i].v_id;
  }

  /* accumulate list of triangles to display */

  for ( j = SDIM ; j < HOMDIM-1 ; j++ ) a[j] = 0.0;
  for ( i = 0 ; i < FACET_VERTS ; i++ )
  {
    for ( j = 0 ; (j < SDIM) && (j < HOMDIM-1) ; j++ ) 
      a[j] = gdata[i].x[j];
    a[HOMDIM-1] = 1.0;
    matvec_mul(view,a,b[i],HOMDIM,HOMDIM);  /* transform */
    if ( SDIM <= 2 )
    { t->x[i][0] = (float)b[i][0];
      t->x[i][1] = (float)b[i][1];
    }
    else
    { t->x[i][0] = (float)b[i][1];
      t->x[i][1] = (float)b[i][2];
      t->x[i][2] = (float)b[i][0];
    }
  } 
  if ( SDIM <= 2 )
  { t->normal[0] = t->normal[1] = 0.0;
    t->normal[2] = 1.0;
  }
  else
  { vnormal(b[0],b[1],b[2],normal); 
    mag = sqrt(SDIM_dot(normal,normal));
    if ( mag > 0.0 )
    { t->normal[0] = (float)(normal[1]/mag); 
      t->normal[1] = (float)(normal[2]/mag); 
      t->normal[2] = (float)(normal[0]/mag);
      if ( fabs(t->normal[2]) < 1e-6 ) t->normal[2] = 0.0; 
    } else
    { t->normal[0] = 0.0f; 
      t->normal[1] = 0.0f; 
      t->normal[2] = 1.0f;
    }
  }
  if ( t->normal[2] > (float)0.0 ) /* frontward normal */
  { int c;
    vertex_id tv;
    for ( i = 0 ; i < gdim ; i++ )
    { float temp = (float)t->x[1][i];
      t->x[1][i] = t->x[2][i];
      t->x[2][i] = temp;
      t->normal[i] = -t->normal[i];
    }
    c = t->ecolor[0];
    t->ecolor[0] = t->ecolor[2];
    t->ecolor[2] = c;
    c = t->etype[0] ^ LABEL_REVERSED;
    t->etype[0] = t->etype[2] ^ LABEL_REVERSED;
    t->etype[2] = (short)c;
    t->etype[1] ^= LABEL_REVERSED;
    t->color = gdata[0].color;
    tv = t->v_id[1];
    t->v_id[1] = t->v_id[2];
    t->v_id[2] = tv;
    t->flag |= FLIPPED_FACET;
  }
  else
  { 
    if ( backcull_flag && (gdata[0].color == gdata[0].backcolor) ) return;
  }

  /* find extents */
  find_bbox(t);

  if ( painter_multiple_sweep_flag == MULTIPLE_SWEEP_PREP )
  { sweep_prep_one(t);
    return;
  }
  else if ( painter_multiple_sweep_flag == MULTIPLE_SWEEP_GO )
  { if ( t->mins[SDIM-1] > sweep_top || t->mins[SDIM-1] <= sweep_bottom ) 
      return;
  }

  if ( setquadcode(t) == OUTOFBOX ) return;

  count++;
} // end painter_facet()

 /* stats for analyzing performance; REAL to handle large counts */
 REAL in_back_calls;
 REAL box_overlaps;
 REAL facetfacet;
 REAL facetedge;
 REAL edgeedge;
 REAL crossings;
 REAL swaps;
 REAL done;
 REAL loopbailouts;
 REAL sep_line_calls;
 REAL sep_plane_calls;

 struct tsort **tlist;
 /*struct tsort *depthhead;*/

/*************************************************************************
*
* function: search_subtree()
*
* purpose: search node and subtree of quadtree depth lists for an
*          element obscured by given element.
* return: pointer to obscured element, or NULL if none found.
*         also retval to indicate type of relationship
*/
struct tsort *search_subtree(
  int qinx, /* index of node in quadtree */
  struct tsort *tk,  /* given element */
  int *retval
)
{ struct tsort *tj;
 
  *retval = 0;

  /* check overall subtree max depth */
  if ( tk->maxs[2] <= qtree[qinx].maxdepth )
    return NULL;

  /* the node itself */
  for ( tj = qtree[qinx].depthhead ; tj != NULL ; tj = tj->next )
  { 
    if ( tj == tk ) continue;
    if ( tk->maxs[2] <= tj->mins[2] ) 
      break; 
    *retval = in_back(tk,tj);  
    if ( *retval & (FIRST_BACK|COPLANAR|DISJOINT) ) continue;
    return tj;
  }

  /* one child */
  tj = search_subtree(2*qinx,tk,retval);
  if ( tj ) return tj;
  
  /* other child */
  return search_subtree(2*qinx+1,tk,retval);
} // end search_subtree()

/**************************************************************************
*
* Function: painter_process_trilist()
*
* Purpose: depth-sort and display trilist.  Separated out for benefit
*          of large surfaces that re-populate trilist in stages.
*/

void painter_process_trilist()
{ size_t k;
  size_t loopcount; /* for emergency loop bailout */
  struct tsort **ll,*tt;
  size_t quadalloc;
  size_t k_top;  /* top of tlist */


  if ( SDIM == 2 )  /* don't bother with depth */
  { for ( k = 0 ; k < count ; k++ )
      visibility_stage(trilist+k);
    return;
  } 

  /* now sort on min z, moving pointers instead of structures */
  /* leaving room at front of list for extra fragments */
  tlist = (struct tsort **)temp_calloc(count+TEXTRA,sizeof(struct tsort *));
  for ( k = 0, ll=tlist+TEXTRA, tt=trilist ; k < count ; k++ ) *(ll++) = tt++;
  qsort((char *)(tlist+TEXTRA),count,sizeof(struct tsort *),FCAST ttcompare); 
  for ( k = 0 ; k < TEXTRA ; k++ ) 
  { tlist[k] = textra+k; tlist[k]->spot = k; tlist[k]->flag = 0; }
  for ( k = TEXTRA ; k < TEXTRA+count ; k++ )
    tlist[k]->spot = k;

  /* quadtree of depth lists */
  maxquaddepth = 8; /* maybe make this adjustable later */
  quadalloc = 2 << (2*maxquaddepth + 1);
  qtree = (struct qtree_t *)temp_calloc(quadalloc,sizeof(struct qtree_t));
  for ( k = 0 ; k < quadalloc ; k++ )
    qtree[k].maxdepth = 1e30f;
  for ( k = count+TEXTRA-1 ; k >= TEXTRA ; k-- )
    qdepth_insert(tlist[k]);

  /* display */
  loopcount = 0;
  k_top = count+TEXTRA;
  for ( k = TEXTRA ; k < k_top ;  )
  { struct tsort *tk = tlist[k];
    struct tsort *tj;
    int sinx,qinx=0;
    int retval;

    if ( breakflag ) break;
    if ( !tk->flag ) { k++; continue; }

    if ( painter_multiple_sweep_flag && tk->maxs[2] > sweep_top )
       return;  // time to get the next sweep

    /* for debugging and testing */
    if ( k > debug_k )
       goto draw_it;

    /* tk is current candidate back facet */

    /* search quadtree list for any z overlap */
    /* First, node to root  */
    qinx = get_quadindex(tk->quadcode);
    for ( sinx = qinx >> 1 ; sinx != 0 ; sinx >>= 1 )
    { for ( tj = qtree[sinx].depthhead ; tj != NULL ; tj = tj->next )
      { 
        if ( tj == tk ) continue;
        if ( tk->maxs[2] <= tj->mins[2] ) 
          break; 
        retval = in_back(tk,tj);  
        if ( retval & (FIRST_BACK|COPLANAR|DISJOINT) ) continue;
        goto have_conflict;
      }
    }

    /* now search subtree for conflicts */ 
    tj = search_subtree(qinx,tk,&retval);
    if ( tj==NULL ) goto draw_it;

have_conflict:
    /* Now have conflict, tk obscuring tj */

      /* test for possible looping, and if found, split tk */
      if ( tj->backstamp == backstamp )
      { int ret;

        crossings++;
        if ( ++loopcount > count ) 
        { loopbailouts++; goto draw_it; }

        /* need to split */ 
               
        if ( k < 2 ) 
        { /* not enough room, so expand tlist allocation, with free at start */
          size_t newsize = 2*k_top;
          size_t n;
          struct tsort *more = (struct tsort*)temp_calloc(k_top,sizeof(struct tsort));
          tlist = (struct tsort**)temp_realloc((char*)tlist,newsize*sizeof(struct tsort*));
          for ( n = 0 ; n < k_top ; n++ )
          { tlist[n+k_top] = tlist[n];
            tlist[n] = more+n;
          }
          for ( n = 0 ; n < newsize ; n++ )
            tlist[n]->spot = n;
          k += k_top;
          k_top = newsize;
        }
        if( retval & BSPLITTINGA )
        {
          ret = newell_split(tk,tj,tlist[k-1],tlist[k-2]);
          if ( ret )
          {
            k -= ret;
            goto repeat_tests;  /* might not have split */
          }
        }
        else if ( retval & ASPLITTINGB )
        {
          /* try splitting the other way */
          ret = newell_split(tj,tk,tlist[k-1],tlist[k-2]);
          if ( ret )
          { k -= ret;
            goto repeat_tests;
          }
        }
        else if ( ((tk->flag & 0xF) == EDGE) && ((tj->flag & 0xF) == EDGE) )
        {
          ret = newell_split(tk,tj,tlist[k-1],tlist[k-2]);
          if ( ret )
          {
            k -= ret;
            goto repeat_tests;  /* might not have split */
          }
        }
      }

      tk->backstamp = backstamp;
      /* swap tj and tk */
      tlist[k] = tj; 
      tlist[tj->spot] = tk; 
      tk->spot = tj->spot;
      tj->spot = k;  
      swaps++;
      goto repeat_tests;

draw_it:
    visibility_stage(tk);
    loopcount = 0;
    tk->flag = 0;   /* to indicate empty structure */

    /* remove from depth list */
    if ( tk == qtree[qinx].depthhead ) 
      qtree[qinx].depthhead = tk->next;
    sinx = qinx;
    /* fix up subtree maxdepths */
    while ( sinx && ( tk->mins[2] <= qtree[sinx].maxdepth ) )
    { float maxd = 1e30f;
      if ( qtree[sinx].depthhead )
         maxd = qtree[sinx].depthhead->mins[2];
      if ( sinx < (1 << (2*maxquaddepth)) )
      { if ( maxd > qtree[2*sinx].maxdepth )
          maxd = qtree[2*sinx].maxdepth;
        if ( maxd > qtree[2*sinx+1].maxdepth )
          maxd = qtree[2*sinx+1].maxdepth;
      }
      qtree[sinx].maxdepth = maxd;
      sinx >>= 1;
    }
    if ( tk->prev ) tk->prev->next = tk->next;
    if ( tk->next ) tk->next->prev = tk->prev;

repeat_tests: 
  
    continue;
  }

  if ( tlist ) temp_free((char *)tlist);
  if ( qtree ) temp_free((char *)qtree);

}  // end painter_process_trilist()

/**************************************************************************
*
* function: painter_end()
*
* purpose: sort and display facets and edges from trilist.
*/


void painter_end()
{
  size_t k;
 
  in_back_calls = box_overlaps = facetfacet = facetedge = edgeedge = 
   crossings = sep_plane_calls = sep_line_calls = 0;
  loopbailouts = 0;

  if ( count > maxcount ) count = maxcount;    /* in case there was excess */

  /* find bounding box */
  if ( need_bounding_box && !painter_multiple_sweep_flag )
  { struct tsort *t;
    bbox_minx = bbox_miny = 1e20;
    bbox_maxx = bbox_maxy = -1e20;
    for ( k = 0, t = trilist ; k < count ; k++,t++ )
    { if ( t->mins[0] < bbox_minx ) bbox_minx = (REAL)t->mins[0];
      if ( t->mins[1] < bbox_miny ) bbox_miny = (REAL)t->mins[1];
      if ( t->maxs[0] > bbox_maxx ) bbox_maxx = (REAL)t->maxs[0];
      if ( t->maxs[1] > bbox_maxy ) bbox_maxy = (REAL)t->maxs[1];
    }
  }

  if ( !painter_multiple_sweep_flag )
    (*init_graphics)();  // print out postscript header

  painter_process_trilist();

  while ( painter_multiple_sweep_flag )
  { reset_multiple_sweep();
    if ( painter_multiple_sweep_flag == MULTIPLE_SWEEP_DONE )
      break;
    raw_generate();
    painter_process_trilist();
  }

  if ( verbose_flag )
  {
    printf("in_back_calls:   %g\n",(DOUBLE)in_back_calls);
    printf("  facetfacet:    %g\n",(DOUBLE)facetfacet);
    printf("  facetedge:     %g\n",(DOUBLE)facetedge);
    printf("  edgeedge:      %g\n",(DOUBLE)edgeedge);
    printf("box_overlaps:    %g\n",(DOUBLE)box_overlaps);
    printf("sep_line_calls:  %g\n",(DOUBLE)sep_line_calls);
    printf("sep_plane_calls: %g\n",(DOUBLE)sep_plane_calls);
    printf("crossings:       %g\n",(DOUBLE)crossings);
    printf("swaps:           %g\n",(DOUBLE)swaps);
    printf("loop bailouts:   %g\n",(DOUBLE)loopbailouts);
  }

  temp_free((char *)trilist); trilist = NULL;

  if ( visibility_test )
    visibility_end();

  (*finish_graphics)();
} /* end old painter_end() */

/*********************************************************************
*
* function: in_back()
*
* purpose: see if one facet or edge obscures another.
*
* returns DISJOINT, FIRST_BACK, SECOND_BACK, ASPLITTINGB, BSPLITTINGA, 
*   or COPLANAR (possibly bitwise OR)
*/
int in_back(
  struct tsort *ta,
  struct tsort *tb
)
{
  int n;
  
  int retval;
   
  if ( verbose_flag )
  {
    in_back_calls++;
    if ( (ta->flag & 0xF) == FACET  )
    { if ( (tb->flag & 0xF) == FACET  )          
        facetfacet++;
      else facetedge++;
    }
    else  
    { if ( (tb->flag & 0xF) == FACET  )          
        facetedge++;
      else edgeedge++;
    }
  }

  /* quick test with quadcodes */
  if ( ((ta->quadcode & tb->quadcode) != ta->quadcode) &&
       ((ta->quadcode & tb->quadcode) != tb->quadcode) )
    return DISJOINT;

  /* test x and y extent overlap */
  for ( n = 0 ; n < 2 ; n++ )
     if ( (tb->maxs[n] <= ta->mins[n]) || (tb->mins[n] >= ta->maxs[n]) )
        return DISJOINT;  
  if ( ta->maxs[2] <= tb->mins[2] ) return FIRST_BACK;

  box_overlaps++;   /* for verbose stats */

  if ( separating_line(ta,tb) == DISJOINT ) return DISJOINT;  

  retval = plane_test(ta,tb);
  if ( retval & (FIRST_BACK|COPLANAR|DISJOINT) ) return retval;

  return retval;
} // end in_back()

/**************************************************************************
*
* function: get_quadindex()
*
* purpose: convert quadcode to index number in quadtree list.
*
*/
int get_quadindex(unsigned int q)
{ int inx,k;

  inx = 1;
  for ( k = 0 ; k < 2*maxquaddepth ; k++, q >>= 2 )
  { int bits = q & 0x3;
    if ( bits == 0 ) return inx;
    inx <<= 1;
    if ( bits == 2 )
      inx++;
  }
  return inx;
} // end get_quadindex()

/**************************************************************************
*
* function: qdepth_insert()
*
* purpose: insert new fragment in proper place in quadtree depth list.
*          For now, crude linear search.
*/
void qdepth_insert(struct tsort *tc)
{ struct tsort *prev,*next;
  int qinx = get_quadindex(tc->quadcode);
  int sinx;

  /* take care of maxdepths of subtrees */
  sinx = qinx;
  while ( sinx && (tc->mins[2] < qtree[sinx].maxdepth) )
  { qtree[sinx].maxdepth = tc->mins[2];
    sinx >>= 1;
  }

  prev = NULL;
  for ( next=qtree[qinx].depthhead; next != NULL; prev=next, next=next->next )
  {
    if ( tc->mins[2] < next->mins[2] )
    { tc->next = next;
      tc->prev = prev;
      if ( prev )
        prev->next = tc;
      else qtree[qinx].depthhead = tc;
      next->prev = tc;
      goto qdepth_insert_exit;
    }
  }
  /* at end of list */
  if ( prev ) prev->next = tc;
  else qtree[qinx].depthhead = tc;
  tc->next = NULL;
  tc->prev = prev;

qdepth_insert_exit:;
} // end qdepth_insert()

/*************************************************************************
*
* function: newell_split()
*
* purpose: split one triangle by plane of another.
*
* return: number of new elements generated.
*/

int newell_split(
  struct tsort *ta,  /* splittee and fragment return */
  struct tsort *tb,  /* splitter */
  struct tsort *tc,  /* fragment return */
  struct tsort *td   /* fragment return */
)
{ 
  int i;
  REAL d0,d1,d2,db;
  int retval;
  size_t tmpspot;

  backstamp++; /* clear loop indications */

  if ( (tb->flag & 0xF) == EDGE )
  { struct tsort *t;
    if ( (ta->flag & 0xF) == EDGE )
    { /* cut first edge in half */
      tmpspot = tc->spot; *tc = *ta; tc->spot = tmpspot;
      for ( i = 0 ; i < gdim ; i++ )
      { float mid = (ta->x[0][i] + ta->x[1][i])/2; 
        if ( ta->x[0][2] > ta->x[1][2] )  /* ta is back half */
        { ta->x[0][i] = mid; tc->x[1][i] = mid; }
        else { ta->x[1][i] = mid; tc->x[0][i] = mid; }
      }
      for ( i = 0 ; i < gdim ; i++ )
      { ta->mins[i] = (ta->x[0][i]<ta->x[1][i]) ? ta->x[0][i] : ta->x[1][i];
        tc->mins[i] = (tc->x[0][i]<tc->x[1][i]) ? tc->x[0][i] : tc->x[1][i];
        ta->maxs[i] = (ta->x[0][i]>ta->x[1][i]) ? ta->x[0][i] : ta->x[1][i];
        tc->maxs[i] = (tc->x[0][i]>tc->x[1][i]) ? tc->x[0][i] : tc->x[1][i];
      }
      /* find_bbox(tc); */
      if ( setquadcode(tc) == INTHEBOX )
      { qdepth_insert(tc);
        return 1;
      }
      else { tc->flag = 0; return 0; }
    }
    t = ta; ta = tb; tb = t; /* swap so edge first */
  }

  if ( (ta->flag & 0xF) == EDGE ) /* tb assumed to be facet */
  {
    db = dotf(tb->normal,tb->x[0],gdim);
    d0 = dotf(tb->normal,ta->x[0],gdim); 
    d1 = dotf(tb->normal,ta->x[1],gdim); 

    /* fill in fragment info */
    tmpspot = tc->spot; *tc = *ta; tc->spot = tmpspot;
    for ( i = 0 ; i < gdim ; i++ )
    { double mid = (((db-d0)*ta->x[1][i] + (d1-db)*ta->x[0][i])/(d1-d0));
      if ( ta->x[0][2] < ta->x[1][2] ) /* keep ta as back part */
        ta->x[1][i] = tc->x[0][i] = (float)mid;
      else  ta->x[0][i] = tc->x[1][i] = (float)mid;
    }
    for ( i = 0 ; i < gdim ; i++ )
    { ta->mins[i] = (ta->x[0][i]<ta->x[1][i]) ? ta->x[0][i] : ta->x[1][i];
      tc->mins[i] = (tc->x[0][i]<tc->x[1][i]) ? tc->x[0][i] : tc->x[1][i];
      ta->maxs[i] = (ta->x[0][i]>ta->x[1][i]) ? ta->x[0][i] : ta->x[1][i];
      tc->maxs[i] = (tc->x[0][i]>tc->x[1][i]) ? tc->x[0][i] : tc->x[1][i];
    }
    /* find_bbox(tc); */
    if ( setquadcode(tc) == INTHEBOX )
    { qdepth_insert(tc);
      return 1;
    }
    else { tc->flag = 0; return 0; }
  }

  /* figure out which vertices of ta on same side, and get as 0,1 */
  db = dotf(tb->normal,tb->x[0],gdim);
  d0 = dotf(tb->normal,ta->x[0],gdim); 
  d1 = dotf(tb->normal,ta->x[1],gdim); 
  d2 = dotf(tb->normal,ta->x[2],gdim); 

if ( (d0<db) && (d1<db) && (d2<db) ) /* should never happen */
{ struct tsort t;  
  t = *ta; *ta = *tb; *tb = t; return 0; }
if ( (d0>db) && (d1>db) && (d2>db) )
{ return 0; }

  retval = 0;
  if ( db == d0 ) /* split thru vertex 0 of ta */
  { 
     tmpspot = tc->spot; *tc = *ta; tc->spot = tmpspot;

     /* fill in fragment info */
     for ( i = 0 ; i < gdim ; i++ )
     { ta->x[2][i] = tc->x[1][i] = 
          (float)(((db-d1)*ta->x[2][i] + (d2-db)*ta->x[1][i])/(d2-d1));
     }
     /* internal edges invisible */
     ta->etype[2] = tc->etype[0] = SPLITTING_EDGE;
     retval = 1;
  }
  if ( db == d1 ) /* split thru vertex 1 of ta */
  { 
     tmpspot = tc->spot; *tc = *ta; tc->spot = tmpspot;

     /* fill in fragment info */
     for ( i = 0 ; i < gdim ; i++ )
     { ta->x[2][i] = tc->x[0][i] = 
          (float)(((db-d0)*ta->x[2][i] + (d2-db)*ta->x[0][i])/(d2-d0));
     }
     /* internal edges invisible */
     ta->etype[1] = tc->etype[0] = SPLITTING_EDGE;
     retval = 1;
  }
  if ( db == d2 ) /* split thru vertex 2 of ta */
  { 
    tmpspot = tc->spot; *tc = *ta; tc->spot = tmpspot;

    /* fill in fragment info */
    for ( i = 0 ; i < gdim ; i++ )
    { ta->x[1][i] = tc->x[0][i] = 
          (float)(((db-d0)*ta->x[1][i] + (d1-db)*ta->x[0][i])/(d1-d0));
    }
    /* internal edges invisible */
    ta->etype[1] = tc->etype[2] = SPLITTING_EDGE;
    retval = 1;
  }

  if ( retval == 1 )
  {
     /* set mins and maxs */
     for ( i = 0 ; i < gdim ; i++ )
     { int j;
        ta->mins[i] = tc->mins[i] = (float)1e30;
        ta->maxs[i] = tc->maxs[i] = (float)(-1e30);
        for ( j = 0 ; j < FACET_VERTS ; j++ )
        { if ( ta->x[j][i] < ta->mins[i] ) ta->mins[i] = ta->x[j][i];
          if ( tc->x[j][i] < tc->mins[i] ) tc->mins[i] = tc->x[j][i];
          if ( ta->x[j][i] > ta->maxs[i] ) ta->maxs[i] = ta->x[j][i];
          if ( tc->x[j][i] > tc->maxs[i] ) tc->maxs[i] = tc->x[j][i];
        }
     }
     if ( ta->mins[2] > tc->mins[2] )
     { struct tsort tmp;  /* get ta as back part */
       tmp = *ta;
       *ta = *tc;
       *tc = tmp;
       tmpspot = ta->spot; ta->spot = tc->spot; tc->spot = tmpspot;
     }
     /* find_bbox(tc); */
     if ( setquadcode(tc) == INTHEBOX )
     { qdepth_insert(tc);
       return 1;
     }
     else { tc->flag = 0; return 0; }
  }

  if ( (d0-db)*(d2-db) > 0.0 )
  { int c = ta->ecolor[1]; 
    short e = ta->etype[1];
    REAL d = d1;
    for ( i = 0 ; i < gdim ; i++ )
    { float temp = ta->x[1][i]; 
      ta->x[1][i] = ta->x[0][i]; ta->x[0][i] = ta->x[2][i];
      ta->x[2][i] = temp;
    }
    ta->ecolor[1] = ta->ecolor[0]; ta->ecolor[0] = ta->ecolor[2];
    ta->ecolor[2] = c;
    ta->etype[1] = ta->etype[0]; ta->etype[0] = ta->etype[2];
    ta->etype[2] = e;
    d1 = d0; d0 = d2; d2 = d;
  }
  else if ( (d1-db)*(d2-db) > 0.0 )
  { int c = ta->ecolor[1]; 
    short e = ta->etype[1];
    REAL d = d1;
    for ( i = 0 ; i < gdim ; i++ )
    { float temp = ta->x[1][i]; 
      ta->x[1][i] = ta->x[2][i]; ta->x[2][i] = ta->x[0][i];
      ta->x[0][i] = temp;
    }
    ta->ecolor[1] = ta->ecolor[2]; ta->ecolor[2] = ta->ecolor[0];
    ta->ecolor[0] = c;
    ta->etype[1] = ta->etype[2]; ta->etype[2] = ta->etype[0];
    ta->etype[0] = e;
    d1 = d2; d2 = d0; d0 = d;
  }


  /* copy all info to fragments */
  tmpspot = tc->spot; *tc = *ta; tc->spot = tmpspot;
  tmpspot = td->spot; *td = *ta; td->spot = tmpspot;
  retval = 2;

  /* fill in fragment info */
  for ( i = 0 ; i < gdim ; i++ )
  { ta->x[2][i] = tc->x[0][i] = td->x[0][i] =
        (float)(((db-d0)*td->x[2][i] + (d2-db)*ta->x[0][i])/(d2-d0));
    tc->x[2][i] = td->x[1][i] =
        (float)(((db-d1)*td->x[2][i] + (d2-db)*ta->x[1][i])/(d2-d1));
  }
  /* internal edges invisible */
  ta->etype[1] = tc->etype[0] = tc->etype[2] = td->etype[0] = SPLITTING_EDGE;

  /* set mins and maxs */
  for ( i = 0 ; i < gdim ; i++ )
  { int j;
    ta->mins[i] = tc->mins[i] = td->mins[i] = (float)1e30;
    ta->maxs[i] = tc->maxs[i] = td->maxs[i] = (float)(-1e30);
    for ( j = 0 ; j < 3 ; j++ )
    { if ( ta->x[j][i] < ta->mins[i] ) ta->mins[i] = ta->x[j][i];
      if ( tc->x[j][i] < tc->mins[i] ) tc->mins[i] = tc->x[j][i];
      if ( td->x[j][i] < td->mins[i] ) td->mins[i] = td->x[j][i];
      if ( ta->x[j][i] > ta->maxs[i] ) ta->maxs[i] = ta->x[j][i];
      if ( tc->x[j][i] > tc->maxs[i] ) tc->maxs[i] = tc->x[j][i];
      if ( td->x[j][i] > td->maxs[i] ) td->maxs[i] = td->x[j][i];
    }
  }

  /* get ta as back part */
  if ( (tc->mins[2] < ta->mins[2]) && (tc->mins[2] <= td->mins[2]) )
  { struct tsort tmp;  /* get ta as back part */
    tmp = *ta;
    *ta = *tc;
    *tc = tmp;
    tmpspot = ta->spot; ta->spot = tc->spot; tc->spot = tmpspot;
  }
  else if ( td->mins[2] < ta->mins[2] )
  { struct tsort tmp;  /* get ta as back part */
    tmp = *ta;
    *ta = *td;
    *td = tmp;
    tmpspot = ta->spot; ta->spot = td->spot; td->spot = tmpspot;
  }

  /* find_bbox(tc); */
  /* find_bbox(td); */
  if ( setquadcode(tc) == INTHEBOX )
  { qdepth_insert(tc);
    if ( setquadcode(td) == INTHEBOX )
      qdepth_insert(td);
    else { retval--; td->flag = 0; td->next = (struct tsort *)0xFF; }
  }
  else
  { /* discard tc */
    tmpspot = tc->spot;
    *tc = *td; 
    tc->spot = tmpspot;
    td->flag = 0; td->next = (struct tsort *)0xAA;
    retval--;
    if ( setquadcode(tc) == INTHEBOX )
      qdepth_insert(tc);
    else { retval--; tc->flag = 0; tc->next = (struct tsort *)0xEE; }
  }
  return retval;
} // end newell_split()

/*********************************************************************
*
* function: separating_plane()
*
* purpose: See if two facets have a separating plane in 3D.
*             Meant to be called when known the facets overlap in 2D.
*
* returns  FIRST_BACK, SECOND_BACK, or ASPLITTINGB | BSPLITTINGA
*/
int separating_plane(
  struct tsort *ta,
  struct tsort *tb,
  int depth /* to limit recursion to depth 2 */
)
{ int i,j;
  int na,nb; /* vertices on respective elements */
  int nna; /* number of vertex pairs to check on ta */
  float *a[3],*b[3];
  int retval;
  REAL da,db,d;
  int n,k,bnear=0,beq=0,bfar=0,anear=0,afar=0,aeq=0;

  sep_plane_calls++;  /* for verbose statistics */

  /* see where tb is with respect to ta plane */
  if ( (ta->flag & 0xF) == FACET  )
  { da = dotf(ta->normal,ta->x[0],gdim);
    n = ((tb->flag & 0xF) == FACET) ? 3 : 2;
    for ( k = 0 ; k < n ; k++ )
    { d = dotf(ta->normal,tb->x[k],gdim); 
      if ( d < da - 0.0001 ) { bnear++; continue; }
      if ( d < da + 0.0001 ) { beq++; continue; }
      bfar++;
    }
    if ( beq == n ) 
      return ((tb->flag&0xF)==EDGE)?FIRST_BACK:DISJOINT; /* in same plane */
    if ( bfar == 0 ) return FIRST_BACK;
  }

  /* see where ta is with respect to tb plane */
  if ( (tb->flag & 0xF) == FACET )
  { db = dotf(tb->normal,tb->x[0],gdim);
    n = ((ta->flag & 0xF) == FACET) ? 3 : 2;
    for ( k = 0 ; k < n ; k++ )
    { d = dotf(tb->normal,ta->x[k],gdim); 
      if ( d < db - 0.0001 ) { anear++; continue; }
      if ( d < db + 0.0001 ) { aeq++; continue; }
      afar++;
    }
    if ( aeq == n ) 
      return ((ta->flag&0xF)==EDGE)?SECOND_BACK:DISJOINT; /* same plane */
     if ( anear == 0 ) return FIRST_BACK;
  }
  
  na = (ta->flag & 0xF) == FACET ? 3 : 2;
  nna = (ta->flag & 0xF) == FACET ? 3 : 1;
  nb = (tb->flag & 0xF) == FACET ? 3 : 2;

  for ( i = 0 ; i < nna ; i++ )
  for ( j = 0 ; j < nb ; j++ )
  { REAL c[3],d; /* coefficients for plane */
    REAL da[3],db[3],minarea,s[3],length,dar[3],dbr[3];
    int ii,jj,jjj;

    for ( ii = 0 ; ii < na ; ii++ )
       a[ii] = ta->x[(i+ii)%na];
    for ( jj = 0 ; jj < nb ; jj++ )
       b[jj] = tb->x[(j+jj)%nb];
    for ( ii = 0 ; ii < 3 ; ii++ ) s[ii] = a[1][ii]-a[0][ii];
       length = sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
    minarea = 1e-4*length;
    c[0] = s[1]*(b[0][2]-a[0][2]) - s[2]*(b[0][1]-a[0][1]);
    c[1] = s[2]*(b[0][0]-a[0][0]) - s[0]*(b[0][2]-a[0][2]);
    c[2] = s[0]*(b[0][1]-a[0][1]) - s[1]*(b[0][0]-a[0][0]);

    if ( c[0]*c[0] + c[1]*c[1] + c[2]*c[2] <= minarea*minarea ) 
         continue; /* degenerate */

    d = c[0]*a[0][0] + c[1]*a[0][1] + c[2]*a[0][2];

    for ( ii = 2 ; ii < na ; ii++ )
    { dar[ii] = c[0]*a[ii][0] + c[1]*a[ii][1] +c[2]*a[ii][2] - d;
      da[ii] = ( dar[ii] < -minarea ? -1.0 : ( dar[ii] > minarea ? 1.0 : 0.0));
    }
    for ( jj = 1 ; jj < nb ; jj++ )
    { dbr[jj] = c[0]*b[jj][0] + c[1]*b[jj][1] +c[2]*b[jj][2] - d;
      db[jj] = ( dbr[jj] < -minarea ? -1.0 : ( dbr[jj] > minarea ? 1.0 : 0.0));
    }

    /* test opposite sidedness */
    for ( jj = 1 ; jj < nb ; jj++ )
      for ( jjj = jj+1 ; jjj < nb ; jjj++ )
        if ( db[jj]*db[jjj] < 0.0 ) goto keeptrying;
    for ( ii = 2 ; ii < na ; ii++ )
      for ( jj = 1 ; jj < nb ; jj++ )
        if ( da[ii]*db[jj] > 0.0 ) goto keeptrying;

     /* have separating plane */
    { REAL asum,bsum;
      /* decide which is in front */
      for ( ii = 2, asum = 0.0 ; ii < na ; ii++ ) asum += c[2]*da[ii];
      for ( jj = 1, bsum = 0.0 ; jj < nb ; jj++ ) bsum += c[2]*db[jj];
      if ( asum > 0.0 || bsum < 0.0 ) return SECOND_BACK;
      else   return FIRST_BACK;
    }
keeptrying: ;
  }

  /* might have case of separating plane having two vertices on tb */
  if ( depth == 2 ) 
      return ASPLITTINGB|BSPLITTINGA;
  retval = separating_plane(tb,ta,2);
  if ( retval == FIRST_BACK ) 
      return SECOND_BACK;
  if ( retval == SECOND_BACK ) 
      return FIRST_BACK;
  return retval;

} // end separating_plane()

/*********************************************************************
*
* function: separating_line()
*
* purpose: See if two elements have a separating line in 2D.
*          To be called after bounding box tests.
*          Includes small tolerance.
*
* returns  DISJOINT or  NOTKNOWN
*/
int separating_line(
  struct tsort *ta,
  struct tsort *tb
)
{ int i;
  int same = 0;
  int na,nb; /* vertices on respective elements */
  int nna,nnb; /* number of lines to try */
  int apos,aneg,bpos,bneg;
  float *a[3],*b[3];
  REAL width; /* thickness of separating line */ 

  sep_line_calls++;  /* for verbose statistics */

  /* get edge first, if any */
  if ( ((ta->flag & 0xF) == FACET) && ((tb->flag & 0xF) == EDGE ) )
  { struct tsort *tmp = ta; ta = tb; tb = tmp; }

  /* want to prevent overlap of facet with thick edge; not going
     to worry about edge-edge overlap, since that too weird. */
  if ( ((ta->flag & 0xF) == EDGE) && ((tb->flag & 0xF) == FACET ) )
  {
    width = ta->width/2;  /* actually need half-width */
  }
  else width = -1e-5;  /* allow slight overlap for numerical purposes */

  na = (ta->flag & 0xF) == FACET ? 3 : 2;
  nb = (tb->flag & 0xF) == FACET ? 3 : 2;
  nna = (ta->flag & 0xF) == FACET ? 3 : 1;
  nnb = (tb->flag & 0xF) == FACET ? 3 : 1;

  /* Try using edges of ta */
  for ( i = 0 ; i < nna ; i++ )
  { REAL cx,cy,d; /* coefficients for line */
    REAL dar[3],dbr[3],minarea;
    int ii,jj;

    for ( ii = 0 ; ii < na ; ii++ )
       a[ii] = ta->x[(i+ii)%na];

    for ( jj = 0 ; jj < nb ; jj++ )
       b[jj] = tb->x[jj];

     cx = a[1][1] - a[0][1]; cy = a[0][0] - a[1][0]; 
     d = cx*a[0][0] + cy*a[0][1];

     minarea = width*sqrt(cx*cx + cy*cy);

     if ( fabs(minarea) < 1e-20 ) 
      { same++;    continue; /* same point */ }

     apos = aneg = bpos = bneg = 0;
     for ( ii = 2 ; ii < na ; ii++ )
     { dar[ii] = cx*a[ii][0] + cy*a[ii][1] - d;
       if ( dar[ii] > minarea ) apos++;
       if ( dar[ii] < -minarea ) aneg++;
     }
     for ( jj = 0 ; jj < nb ; jj++ )
     { dbr[jj] = cx*b[jj][0] + cy*b[jj][1] - d;
       if ( dbr[jj] > minarea ) bpos++;
       if ( dbr[jj] < -minarea ) bneg++;
     }

     /* test opposite sidedness */
     if ( apos == (na-2) && bneg == nb ) return DISJOINT;
     if ( aneg == (na-2) && bpos == nb ) return DISJOINT;
  }

  /* Try using edges of tb */
  for ( i = 0 ; i < nnb ; i++ )
  { REAL cx,cy,d; /* coefficients for line */
    REAL dar[3],dbr[3],minarea;
    int ii,jj;

    for ( ii = 0 ; ii < nb ; ii++ )
       a[ii] = tb->x[(i+ii)%nb];

    for ( jj = 0 ; jj < na ; jj++ )
       b[jj] = ta->x[jj];

     cx = a[1][1] - a[0][1]; cy = a[0][0] - a[1][0]; 
     d = cx*a[0][0] + cy*a[0][1];

     width = -1e-5;
     minarea = width*sqrt(cx*cx + cy*cy);

     if ( fabs(minarea) < 1e-20 ) 
      { same++;    continue; /* same point */ }

     apos = aneg = bpos = bneg = 0;
     for ( ii = 2 ; ii < nb ; ii++ )
     { dar[ii] = cx*a[ii][0] + cy*a[ii][1] - d;
       if ( dar[ii] > minarea ) apos++;
       if ( dar[ii] < -minarea ) aneg++;
     }
     for ( jj = 0 ; jj < na ; jj++ )
     { dbr[jj] = cx*b[jj][0] + cy*b[jj][1] - d;
       if ( dbr[jj] > minarea ) bpos++;
       if ( dbr[jj] < -minarea ) bneg++;
     }

     /* test opposite sidedness */
     if ( apos == (nb-2) && bneg == na ) return DISJOINT;
     if ( aneg == (nb-2) && bpos == na ) return DISJOINT;
  }

  return NOTKNOWN;
} // end separating_line(

/*********************************************************************
*
* function: plane_test()
*
* purpose: See if one facet or edge is in front or back of element plane.
*          Suitable for Newell-Newell-Sancha algorithm.
*
* returns DISJOINT, FIRST_BACK, SECOND_BACK, ASPLITTINGB, BSPLITTINGA, or COPLANAR 
* Returns FIRST_BACK if guaranteed first does not obscure any of second.
*  Possibly bitwise OR of properties.
*/
int plane_test(
  struct tsort *ta,
  struct tsort *tb
)
{
  REAL da,db;
  int k,n;
  int afar=0,aeq=0,anear=0;  /* count of ta vertices relative to tb plane */
  int bfar=0,beq=0,bnear=0;  /* count of tb vertices relative to ta plane */
  REAL d;
  int retval = NOTKNOWN;

  /* if two edges */
  if ( ((ta->flag & 0xF) == EDGE) && ((tb->flag & 0xF) == EDGE) )
  { REAL ab1x = tb->x[0][0] - ta->x[0][0];
    REAL ab1y = tb->x[0][1] - ta->x[0][1];
    REAL ab1z = tb->x[0][2] - ta->x[0][2];
    REAL ab2x = tb->x[1][0] - ta->x[0][0];
    REAL ab2y = tb->x[1][1] - ta->x[0][1];
    REAL ab2z = tb->x[1][2] - ta->x[0][2];
    REAL aax  = ta->x[1][0] - ta->x[0][0];
    REAL aay  = ta->x[1][1] - ta->x[0][1];
    REAL aaz  = ta->x[1][2] - ta->x[0][2];
    REAL area = ab1x*ab2y - ab1y*ab2x;
    REAL vol = (ab1x*ab2y - ab1y*ab2x)*aaz + (ab1y*ab2z - ab1z*ab2y)*aax
              + (ab1z*ab2x - ab1x*ab2z)*aay;
    if ( vol == 0.0 ) return COPLANAR;
    if ( area == 0.0 ) return DISJOINT;
    if ( area*vol > 0 ) return SECOND_BACK;
    return FIRST_BACK;
  }

  /* see where tb is with respect to ta plane */
  da = dotf(ta->normal,ta->x[0],gdim);
  n = ((tb->flag & 0xF) == FACET) ? 3 : 2;
  for ( k = 0 ; k < n ; k++ )
  {
    d = dotf(ta->normal,tb->x[k],gdim); 
    if ( d < da - 0.0001 ) { bnear++; continue; }
    if ( d < da + 0.0001 ) { beq++; continue; }
    bfar++;
  }
  if ( beq == n ) return COPLANAR; /* both in same plane */
  if ( bfar == 0 ) return FIRST_BACK; 
  if ( bnear > 0 ) 
  { if ( (ta->flag & 0xF) == FACET ) retval = ASPLITTINGB; }
  else retval = SECOND_BACK;
  
  /* see where ta is with respect to tb plane */

  db = dotf(tb->normal,tb->x[0],gdim);
  n = ((ta->flag & 0xF) == FACET) ? 3 : 2;
  for ( k = 0 ; k < n ; k++ )
  {
    d = dotf(tb->normal,ta->x[k],gdim); 
    if ( d < db - 0.0001 ) { anear++; continue; }
    if ( d < db + 0.0001 ) { aeq++; continue; }
    afar++;
  }
  if ( aeq == n ) return COPLANAR; /* both in same plane */
  if ( anear == 0 ) return FIRST_BACK;
  if ( afar > 0 )  
  { if ( (tb->flag & 0xf) == FACET ) retval |= BSPLITTINGA;   }
  else retval |= SECOND_BACK;
  
  /* might still not have properly detected order, so try this */
  /*
  if ( !(retval & (FIRST_BACK|SECOND_BACK)) )
    retval = separating_plane(ta,tb,0);
    */
    
  return retval;

} /* end plane_test() */

/************************************************************************
 ************************************************************************

 Visibility testing.  Takes output of depth sort and deletes hidden
 elements.  Uses sweep line (at random angle) to track topology.
 Much like Stuart Sechrest and Donald P. Greenberg, A Visible Polygon
 Reconstruction Algorithm, ACM Transactions on Graphics, vol 1, no 1,
 Jan 1982, pp 25-42.

 General strategy remarks:

 The image is broken down into individual polygon edges.  Each edge is
 "above" or "below" a polygon.  For now, each polygon is just a facet,
 but this could change if thick edges were added in the form of rectangles.

 The algorithm proceeds by moving a sweep line across the image, keeping
 track of "active edges" that intersect the sweep line.  The active edge
 list is altered at "events": edge starts, edge ends, and edge crossings.
 The sweep line is tilted at an arbitrary angle to prevent degenerate
 vertices, except vertices coincident in projection.  Pre-ordered event
 lists are kept for edge starts and ends, and a heap for upcoming
 crossings.

 Attached to each active edge is a list of layers of facets in the area
 immediately above it.

*************************************************************************/
int visdebuglevel;
#define VIS_TIMING  1
#define VIS_LAYERCHECK  2
#define VIS_EVENTDUMP 3
struct vis_conedge;
struct vis_vertex;

/* Heap for ordering upcoming crossing events. */
int vis_heap_count;  /* heap spots used */
int vis_heap_max;    /* heap spots allocated */
struct vis_event *vis_heap;
void vis_insert_heap(struct vis_event *);
void vis_delete_heap(int);
void find_next_event(struct vis_conedge *);
void find_next_event2( struct vis_conedge *, struct vis_conedge *);
void vis_crossing(struct vis_conedge *,struct vis_conedge *); 
int vecount;  /* number of edges in edge list */
int vis_crossing_count; /* just for info */
int add_layer(struct vis_conedge *, struct tsort *);
int delete_layer(struct vis_conedge *, struct tsort *);
int visibility_recover();

int vvcomp( struct vis_vertex *, struct vis_vertex *);
REAL activate_edge( struct vis_conedge *);
void deactivate_edge(struct vis_conedge *);
int handle_vertex_event(struct vis_vertex *);

/* Edge list, three for each facet */
#define MAXLAYERS 20
struct vis_rawedge { struct vis_vertex *v[2]; /*  endpoints */
                     struct tsort *tsrt; /* facet it borders */
                     struct vis_conedge *conedge;
                     int flags;    /* see below */
  };
struct vis_rawedge *vis_rawedges;
struct vis_rawedge **rawplist;  /* pointers for sorting */

int vecomp(struct vis_rawedge **, struct vis_rawedge **);
/* vis_rawedge flag bits */
#define V_FACET_BOTTOM 1
#define V_FACET_TOP   2
#define V_FACET_LEFT   4
#define V_FACET_RIGHT   8
#define V_LAYER_CHECK  0x10

/* Consolidated vertices */
#define MAXVEDGES 25
struct vis_vertex { 
                    REAL x[2];   /* u, v */                   
                    struct vis_vertex **fixup[2];  /* pre-cons reverse pointer */
                    struct vis_conedge *left_edge_head; // linked list
                    struct vis_conedge *right_edge_head;
    };
struct vis_vertex *vis_vertices;
size_t vis_vertex_max;
int vis_vertex_count;

/* Consolidated edges */
struct vis_conedge {
                  int id;    // for debugging
                  struct vis_vertex *v[2];  /* endpoints */
                  REAL m;     /* line slope */
                  int rawstart;  /* associated raw edge start */
                  int rawend;  /* last associated raw edge */
                  struct vis_conedge *prev_active;  /* active list pointer */
                  struct vis_conedge *next_active;  /* active list pointer */
                  struct vis_conedge *left_edge_next; // linked list for left edges of right vertex
                  struct vis_conedge *right_edge_next; // linked list for right edges of left vertex
                  int flags;    /* see below */
                  int layers;  /* number of layers above edge */
                  struct tsort **layer; /* facets "above" edge */
                  int maxlayers; /* allocated space */
                  int seqno;    /* sequence number, for debugging */
    };
struct vis_conedge *vis_conedges;
int vis_conedge_max;
int vis_conedge_count;

/* Crossing event structure */
struct vis_event { REAL time;  /* sweep time of event */
                   struct vis_conedge *e1;  // lower edge on left
                   struct vis_conedge *e2;  // higher edge on left
 };
int vis_event_comp (struct vis_event *, struct vis_event *);
int vertex_event_spot;  // the next event

int wrong_middles;  /* for some debugging */

/* Margin to shorten edge ends, so don't get spurious crossings */
REAL veps = 1e-14;

/* Active edge list */
struct vis_conedge *active_edge_first;
struct vis_conedge sentinel; 
struct vis_vertex sentinelv[4]; /* for sentinel endpoints */

/* List of edges to check for top layer */
struct vis_conedge **check_list;
int check_list_count;
int check_list_max;
void check_visible ();
void check_one_visible (struct vis_conedge *);

/* random rotation coefficients for sweep line */
//REAL vis_va = 0.8432848996472634;
//REAL vis_vb = 0.5869487870825054;
REAL vis_va = 0.99995; // for debugging
REAL vis_vb = 0.01;

REAL sweep_u;  /* current sweep position */

/************************************************************************
*
* function: visibility_stage()
*
* purpose: Adds element to list for visibility testing.
*          Enabled by visibility_test toggle.
*
*/

void visibility_stage(struct tsort *t)
{

  if ( !visibility_test )
  { if ( (t->flag & 0xF) == FACET )
     (*display_facet)(t);
    else (*display_edge)(t);
    return;
  }

  /* accumulate */
  if ( vis_list == NULL )
  { vis_max = maxcount;
    vis_list = (struct tsort *)temp_calloc(vis_max,sizeof(struct tsort));
  }
  else if ( vis_count >= vis_max - 1 )
  { vis_list = (struct tsort *)temp_realloc((char*)vis_list,
        2*vis_max*sizeof(struct tsort));
    vis_max *= 2;
  }

  vis_list[vis_count++] = *t;

} // end visibility_stage()

/*
   FOR DEBUGGING
*/
void active_list_check(void);

void active_list_check()
{ struct vis_conedge *e;
  
  for ( e = active_edge_first ; e != &sentinel ; e = e->next_active )
  { if ( e->next_active->prev_active != e )
      kb_error(2418,"Visibility edge active list bad.\n",RECOVERABLE);
  }
} // end active_list_check())

void print_active(int id)
{ struct vis_conedge *ee;
  int inrow = 0;
  for ( ee = active_edge_first ; ee != &sentinel ; ee = ee->next_active )
  { if ( ee->id == id )
    { printf("\n    ---- %d ---- \n",ee->id);
      inrow = 0;
    }
    else
      printf("%d ",ee->id);
    inrow++;
    if ( inrow==10 )
    { printf("\n");
      inrow = 0;
    }
  }
}

// prints active list 5 edges each way
void print_part_list(struct vis_conedge *e)
{ int k;
  struct vis_conedge *ee;
  for ( ee = e, k = 0 ; ee && k < 20 ; k++, ee = ee->prev_active ) ;
  if ( ee == NULL ) ee = active_edge_first;
  for ( k = 0 ; ee && k <= 40 ; k++, ee=ee->next_active )
  { if ( ee == e ) printf(" -- %d -- ",ee->id);
    else printf("%d ",ee->id);
  }
  printf("\n");
}
/************************************************************************
* 
* function: vvcomp()
*
* purpose: comparison of vertices, for consolidation
*
*/
int vvcomp(
   struct vis_vertex *a,
   struct vis_vertex *b
)
{
  if ( a->x[0] < b->x[0] ) return -1;
  if ( a->x[0] > b->x[0] ) return  1;
  if ( a->x[1] < b->x[1] ) return -1;
  if ( a->x[1] > b->x[1] ) return  1;
  return 0;
} // end int vvcomp()

/************************************************************************
* 
* function: vecomp()
*
* purpose: comparison of raw edges, for consolidation
*
*/
int vecomp(
  struct vis_rawedge **a,
  struct vis_rawedge **b
)
{
  if ( a[0]->v[0] < b[0]->v[0] ) return -1;
  if ( a[0]->v[0] > b[0]->v[0] ) return  1;
  if ( a[0]->v[1] < b[0]->v[1] ) return -1;
  if ( a[0]->v[1] > b[0]->v[1] ) return  1;
  return 0;
} // end int vecomp()

/************************************************************************
*
* function: visibility_lists()
*
* purpose:   Populate raw edge and vertex lists.
*
*/   
void visibility_lists()
{ int tops,bottoms,lefts,rights;
  struct vis_vertex *vv;
  struct vis_rawedge *ve;
  struct tsort *t;
  int i,ii,iii;
  size_t k;

  /* Populate raw edge and vertex lists */
  vis_vertex_max = 3*vis_count;
  vis_vertices = (struct vis_vertex *)temp_calloc(vis_vertex_max,
                                              sizeof(struct vis_vertex));
  vv = vis_vertices;
  vis_vertex_count = 0;
  
  vis_rawedges = (struct vis_rawedge *)temp_calloc(3*vis_count,
                                              sizeof(struct vis_rawedge));
  ve = vis_rawedges;
  vecount = 0;

  for ( k = 0, t = vis_list ; k < vis_count ; k++,t++ )
  { if ( (t->flag & 0xF) == FACET )
    {
      REAL minu = 1e30, maxu = -1e30;

      if ( t->color == CLEAR )
      { t->flag |= VISIBLE; continue; }  /* kludge for now */

      /* rotate coordinates for general position and get x bounding box */

      for ( i = 0 ; i < FACET_VERTS ; i++ )
      { 
        vv[i].x[0] = vis_va*t->x[i][0] + vis_vb*t->x[i][1];
        vv[i].x[1] = -vis_vb*t->x[i][0] + vis_va*t->x[i][1];
        if ( vv[i].x[0] < minu ) minu = vv[i].x[0];
        if ( vv[i].x[0] > maxu ) maxu = vv[i].x[0];
      }

      /* now, the edges */
      tops = 0; bottoms = 0; lefts = 0; rights = 0;
      for ( i = 0 ; i < FACET_VERTS ; i++ )
      { REAL area;
        ii = (i+1 >= FACET_VERTS) ? 0 : i+1;
        if ( vv[i].x[0] <= vv[ii].x[0] )  /* leftmost vertex first */
        { ve->v[0] = vv+i;
          ve->v[1] = vv+ii;
          vv[i].fixup[0] = &ve->v[0];  /* so can adjust edges after sorting vertices */
          vv[ii].fixup[1] = &ve->v[1];
        } else
        { ve->v[0] = vv+ii;
          ve->v[1] = vv+i;
          vv[i].fixup[0] = &ve->v[1];  /* so can adjust edges after sorting vertices */
          vv[ii].fixup[1] = &ve->v[0];
        }

        iii = (ii+1 >= FACET_VERTS) ? 0 : ii+1;  /* third vertex */
        area = (ve->v[1]->x[0]-ve->v[0]->x[0])*(vv[iii].x[1]-ve->v[0]->x[1])
          -(vv[iii].x[0]-ve->v[0]->x[0])*(ve->v[1]->x[1]-ve->v[0]->x[1]); 
        if ( fabs(area) < 1e-14 ) break;

        if ( area > 0 )
        { ve->flags |= V_FACET_BOTTOM; bottoms++; }
        else { ve->flags |= V_FACET_TOP; tops++; }

        if ( ve->v[0]->x[0] == minu )
        { ve->flags |= V_FACET_LEFT; lefts++; }

        if ( ve->v[1]->x[0] == maxu )
        { ve->flags |= V_FACET_RIGHT; rights++; }

        ve->tsrt = t;
        ve++; vecount++;
      }

      /* check we successfully found things, and skip edge-on facets */
      if ( (tops==0) || (bottoms==0) || (lefts!=2) || (rights!=2) )
        continue;

      vv += FACET_VERTS; vis_vertex_count += FACET_VERTS;
    } 
    else  /* lone edge */
    { 
      /* for now, make all lone edges visible */
      t->flag |= VISIBLE;
    }
  }

}  // end visibility_lists()

/************************************************************************
*
* function: visibility_vertex_consolidate()
*
* purpose: Consolidation of vertices, and sorting in u order .
*
*/
void visibility_vertex_consolidate()
{int i,j;
  qsort(vis_vertices,vis_vertex_count,sizeof(struct vis_vertex),FCAST vvcomp);
  for ( i = -1, j = 0 ; j < vis_vertex_count ; j++ )
  { int m;
    if ( (i < 0) || (vvcomp(vis_vertices+i,vis_vertices+j) != 0) ) 
    { vis_vertices[++i] = vis_vertices[j];
    }
    /* use reverse pointers to fix up edges */
    for ( m = 0 ; m < 2 ; m++ )
     *(vis_vertices[j].fixup[m]) = vis_vertices+i;
  }
  vis_vertex_count = i+1;
} // end visibility_vertex_consolidate()

/************************************************************************
*
* function: visibility_edge_consolidate()
*
* purpose: Consolidation of edges.
*
*/
void visibility_edge_consolidate() 
{ int i,j;

  rawplist = (struct vis_rawedge **)temp_calloc(vecount,
                  sizeof(struct vis_rawedge *));
  for ( i = 0 ; i < vecount ; i++ ) rawplist[i] = vis_rawedges+i;
  qsort(rawplist,vecount,sizeof(struct vis_rawedge*),FCAST vecomp);
  vis_conedge_max = vecount;
  vis_conedges = (struct vis_conedge *) temp_calloc(vis_conedge_max,
                     sizeof(struct vis_conedge) );
  for ( i = -1, j = 0 ; j < vecount ; j++ )
  { if ( (j==0) || (vis_conedges[i].v[0] != rawplist[j]->v[0]) || 
         (vis_conedges[i].v[1] != rawplist[j]->v[1]) )
    { // new consolidated edge
      i++;
      vis_conedges[i].id = i;
      vis_conedges[i].v[0] = rawplist[j]->v[0];
      vis_conedges[i].v[1] = rawplist[j]->v[1];
      vis_conedges[i].rawstart = j;
      vis_conedges[i].rawend = j;
    }
    else // same consolidated edge
      vis_conedges[i].rawend++;
    rawplist[j]->conedge = vis_conedges+i;
  }
  i++;
  vis_conedges = (struct vis_conedge*)temp_realloc((char*)vis_conedges,
     i*sizeof(struct vis_conedge));
  vis_conedge_max = vis_conedge_count = i;

} // end visibility_edge_consolidate()

/**********************************************************************
*
* Function: visibility_edge_sort()
*
* Purpose: Sort left edges and right edges at vertices, by slope
*/

void visibility_edge_sort()
{ int i;
  struct vis_conedge *vc,*ee,*prev_ee;

  /* Initialize edge slopes */
  for ( i = 0, vc = vis_conedges; i < vis_conedge_count; i++, vc++ )
  { REAL du = vc->v[1]->x[0] - vc->v[0]->x[0];
    REAL dv = vc->v[1]->x[1] - vc->v[0]->x[1];
    vc->m = dv/du;

    // Install in left vertex right edges, by increasing slope
    prev_ee = NULL;
    for ( ee = vc->v[0]->right_edge_head ; ee ; ee = ee->right_edge_next )
    { if ( vc->m < ee->m )
        break;
      prev_ee = ee;
    }
    vc->right_edge_next = ee;
    if ( prev_ee )
      prev_ee->right_edge_next = vc;
    else 
      vc->v[0]->right_edge_head = vc;

    // Install in right vertex left edges, by decreasing slope
    prev_ee = NULL;
    for ( ee = vc->v[1]->left_edge_head ; ee ; ee = ee->left_edge_next )
    { if ( vc->m > ee->m )
        break;
      prev_ee = ee;
    }
    vc->left_edge_next = ee;
    if ( prev_ee )
      prev_ee->left_edge_next = vc;
    else 
      vc->v[1]->left_edge_head = vc;
  }

} // end visibility_edge_sort()

/************************************************************************
*
* function: visibility_end()
*
* purpose: Run visibility algorithm after accumulation of data.
*
*/
int debug_seq = 0;  /* for debugging */

void visibility_end()
{ 
  size_t k;
  struct tsort *t;

  REAL next_u;
  int vis_display_count;

  int badflag = 0;

  debug_seq = 0;

  /* List of edges to check top facet for */
  check_list_max = 1000;
  check_list = (struct vis_conedge **)temp_calloc(check_list_max,
                   sizeof(struct vis_conedge *));
  check_list_count = 0;

  /* populate raw vertex and edge lists */
  visibility_lists();

  if ( vis_vertex_count == 0 ) 
    goto draw_visible;

  /* Consolidation of vertices and edges */
  visibility_vertex_consolidate();  
  visibility_edge_consolidate();

  if ( badflag ) goto bail_out;

  visibility_edge_sort();

  /* Initialize crossing event heap with sentinel */
  vis_heap_max = vecount > 100 ? vecount : 100;
  vis_heap = (struct vis_event *)temp_calloc(vis_heap_max,
                                          sizeof(struct vis_event));
  vis_heap[0].time = 1e30;
  vis_heap_count = 1;
  vis_crossing_count = 0;

  /* Initialize active edge list with sentinel vertex at the top */
  sentinel.v[0] = sentinelv;
  sentinel.v[1] = sentinelv+1;
  sentinelv[0].x[0] = -1e20;
  sentinelv[0].x[1] = 1e20;
  sentinelv[1].x[0] = 1e20;
  sentinelv[1].x[1] = 1e20;
  sentinel.m = 0;
  sentinel.maxlayers = 0;
  sentinel.layers = 0;
  sentinel.layer = NULL;
  sentinel.prev_active = NULL;
  sentinel.next_active = NULL;
  sentinel.rawend = -1;  // no facets
  active_edge_first = &sentinel;


  /* Sweep */
  vertex_event_spot = 0;
  sweep_u = -1e20;
  while ( vertex_event_spot < vis_vertex_count )
  {
    int retval; /* return code from event handlers; < 0 for error */

    debug_seq++;  // for debugging 

    /* find which is next event and process */
    if ( (vis_heap_count <= 0) || (vis_vertices[vertex_event_spot].x[0] < vis_heap[0].time) )
    { /* do vertex event */
      next_u = vis_vertices[vertex_event_spot].x[0];
      if ( (next_u - sweep_u) > 1e-10 )
         check_visible((next_u+sweep_u)/2); 
      sweep_u = next_u;
      retval = handle_vertex_event(vis_vertices+vertex_event_spot);
      if ( retval < 0 )
      { retval = visibility_recover();
        if ( retval < 0 )
          goto bail_out;
      }
      vertex_event_spot++;
    }
    else /* do crossing event */
    { struct vis_conedge *e1,*e2;
      next_u = vis_heap[0].time;

      if ( (next_u - sweep_u) > 1e-10 )
         check_visible((next_u+sweep_u)/2); 
      e1 = vis_heap[0].e1; 
      e2 = vis_heap[0].e2;
      vis_delete_heap(0);
      if ( visdebuglevel >= VIS_EVENTDUMP )
        printf("crossing %d %d  at %20.15f\n",(int)(e1-vis_conedges),  
              (int)(e2-vis_conedges),(DOUBLE)next_u);
      sweep_u = next_u;   
      vis_crossing(e1,e2);
    }
  } // end while


  goto draw_visible;

bail_out: /* error handling: mark all as visible */
  kb_error(5555,"Abandoning visibility test and drawing all facets.\n",WARNING);
  for ( k = 0, t = vis_list, vis_display_count = 0 ; k < vis_count ; k++,t++ )
    t->flag |= VISIBLE;

draw_visible:
  /* Display elements marked visible */
  for ( k = 0, t = vis_list, vis_display_count = 0 ; k < vis_count ; k++,t++ )
  if ( t->flag & VISIBLE )
  { if ( (t->flag & 0xF) == FACET )
     (*display_facet)(t);
    else (*display_edge)(t);
    vis_display_count++;
  }
  temp_free((char*)vis_heap);  vis_heap = 0;
  temp_free((char*)vis_conedges); vis_conedges = 0;
  temp_free((char*)vis_rawedges); vis_rawedges = 0;
  temp_free((char*)vis_vertices); vis_vertices = 0;
  temp_free((char*)check_list);  check_list = 0;
  temp_free((char*)rawplist);  rawplist = 0;
  temp_free((char*)vis_list);  vis_list = 0;

if (visdebuglevel >= VIS_TIMING)
{
fprintf(stderr,"Visible: %d facets out of %d\n",vis_display_count,(int)vis_count);
fprintf(stderr,"Crossing count: %d\n",vis_crossing_count);
fprintf(stderr,"Wrong middles: %d\n",wrong_middles); wrong_middles = 0;
}
} // end visibility_end()

/************************************************************************
*
* function: vis_event_comp()
*
* purpose: compare times of two events.  Sorts on time, then type,
*          then average slope, then edges.
*/

int vis_event_comp(
  struct vis_event *a, 
  struct vis_event *b
)
{ REAL ma,mb;
  if ( a->time < b->time ) return -1;
  if ( a->time > b->time ) return  1;
  ma = (a->e1->m+a->e2->m);
  mb = (b->e1->m+b->e2->m);
  if ( ma < mb ) return -1;
  if ( ma > mb ) return  1;
  if ( a->e1 < b->e1 ) return -1;
  if ( a->e1 > b->e1 ) return  1;
  if ( a->e2 < b->e2 ) return -1;
  if ( a->e2 > b->e2 ) return  1;
  return 0;
} // end vis_event_comp()

/************************************************************************
*
* function: vis_insert_heap()
*
* purpose: Add edge event to heap list.  Not detecting duplicates;
*          leaving that to validity test when crossing is handled.
*
*/

void vis_insert_heap(struct vis_event *e)
{ int k,kk;
  int result;

  if ( vis_heap_count >= vis_heap_max-1 )
  { vis_heap = (struct vis_event *)kb_realloc((char*)vis_heap,
                 2*vis_heap_max*sizeof(struct vis_event));
    vis_heap_max *= 2;
  }

  for ( k = vis_heap_count ; k > 0 ; k = kk )
  { 
    kk = (k-1)/2;
    result = vis_event_comp(e,&vis_heap[kk]); 
    if ( result < 0 )
    { 
      vis_heap[k] = vis_heap[kk];
    }
    else break;
  }
  vis_heap[k] = *e;
  vis_heap_count++;

} // end vis_insert_heap()

/***************************************************************************
*
* function: vis_delete_heap()
*
* purpose: Delete element n of event heap and adjust heap.
*/

void vis_delete_heap(int n)
{ int k,kk;
  int result;
  struct vis_event e;

  if ( n == vis_heap_count-1 ) 
  { vis_heap_count--;
    goto vis_delete_heap_exit;
  }

  if ( vis_heap_count == 1 )
   kb_error(2421,"vis_delete_heap trying to delete sentinel.\n",RECOVERABLE);

  /* replace with top event */
  e = vis_heap[--vis_heap_count]; 

  /* check direction to percolate */
  result = (n==0) ? 1 : vis_event_comp(&e,vis_heap+(n-1)/2);
  if ( result < 0 )
  { /* downward */
    k = n;
    kk = (n-1)/2;
    vis_heap[k] = vis_heap[kk];
    for ( k = kk ; k > 0 ; k = kk )
    { kk = (k-1)/2;
      result = vis_event_comp(&e,vis_heap+kk);
      if ( result < 0 )
      { vis_heap[k] = vis_heap[kk];
        continue;
      }
      else 
      { break;
      }
    }
    vis_heap[k] = e;
    goto vis_delete_heap_exit; 
  }
  else
  { /* upward */
    for ( k = n; 2*k+1 < vis_heap_count ; k = kk )
    { 
      if ( 2*k+2 >= vis_heap_count )
      { /* only one parent */
        kk = 2*k+1;
        result = vis_event_comp(&e,vis_heap+kk);
        if ( result > 0 )
        { vis_heap[k] = vis_heap[kk]; 
          vis_heap[kk] = e;
          goto vis_delete_heap_exit;
        }
        else 
        { vis_heap[k] = e;
          goto vis_delete_heap_exit;
        }
      }
      else
      { /* two parents */
        result = vis_event_comp(&vis_heap[2*k+1],&vis_heap[2*k+2]);
        kk = (result < 0) ? 2*k+1 : 2*k+2;
        result = vis_event_comp(&e,vis_heap+kk);
        if ( result < 0 )
        { /* done */
          vis_heap[k] = e;
          goto vis_delete_heap_exit;
        }
        else if ( result > 0 )
        { /* keep going up */
          vis_heap[k] = vis_heap[kk];
          continue;
        }
      }
    }
    vis_heap[k] = e;   /* in case at top of heap */
  }
vis_delete_heap_exit: ;

} // end vis_delete_heap()

/****************************************************************************
*
* function: add_layer()
*
* purpose: Add a facet to the layers above an active edge.
*
* return value: 1 if added, 0 if already there.
*/

int add_layer(
  struct vis_conedge *ee,
  struct tsort *f
)
{ int i;
  int retval;

  debug_seq++;

  f->flag |= VISIBILITY_LIVE;
  for ( i = 0 ; i < ee->layers ; i++ )
    if ( ee->layer[i] == f ) break;
  if ( i == ee->maxlayers )
  { int newcount = (ee->maxlayers > 10) ? 2*ee->maxlayers : ee->maxlayers + 10;
    ee->layer = (struct tsort **)temp_realloc((char*)(ee->layer),
        newcount*sizeof(struct tsort*));
    ee->maxlayers = newcount;
  }
  if ( i == ee->layers )
  { /* not already found in layer list */
    ee->layer[ee->layers++] = f;
    if ( visdebuglevel >= VIS_EVENTDUMP )
      fprintf(stderr,"Adding facet %d to edge %d layers.\n",
         (int)(f->f_id & 0xFFFFFF)+1,(int)(ee-vis_conedges));
    if ( !(ee->flags & V_LAYER_CHECK) )
    { 
      if ( check_list_count >= check_list_max )
      { check_list = (struct vis_conedge **)temp_realloc((char*)check_list,
          2*check_list_max*sizeof(struct vis_conedge *));
        check_list_max *= 2;
      }
      check_list[check_list_count++] = ee;
      ee->flags |= V_LAYER_CHECK;
    }
    retval = 1;
  }
  else retval = 0;

  return retval;
} // end add_layer()

/****************************************************************************
*
* function: delete_layer()
*
* purpose: Delete a facet from the layers above an active edge.
*          Not keeping remaining facets in depth order.
* 
* return value: 1 if found, 0 if not.
*/

int delete_layer(
  struct vis_conedge *ee,
  struct tsort *f
)
{ int i;
  int retval = 0;

  for ( i = 0 ; i < ee->layers ; i++ )
    if ( ee->layer[i] == f )
    { ee->layer[i] = ee->layer[--ee->layers];
      if ( visdebuglevel >= VIS_EVENTDUMP )
        fprintf(stderr,"Deleting facet %d from edge %d layers.\n",
          (int)(f->f_id & 0xFFFFF)+1,(int)(ee-vis_conedges));
      if ( !(ee->flags & V_LAYER_CHECK) )
      { if ( check_list_count >= check_list_max )
        { check_list = (struct vis_conedge **)temp_realloc((char*)check_list,
            2*check_list_max*sizeof(struct vis_conedge *));
          check_list_max *= 2;
        }
        check_list[check_list_count++] = ee;
        ee->flags |= V_LAYER_CHECK;
      }
      retval = 1;
      break;
    }

  return retval;
} // end delete_layer()



/*************************************************************************
*
* function: find_next_event()
*
* purpose:  Check for upcoming crossing event.  Doesn't count crossing
*           near end of edge.   Need to do very robust crossing
*           calculation, so not to be fooled by numerical glitches.
*
*/

void find_next_event(struct vis_conedge *e)
{ struct vis_event ev;
  ev.time = 1e30;
 
  /* Above */
  if ( (e->m > e->next_active->m ) && (e->v[1] != e->next_active->v[1]) )
  { REAL u;

    u = e->v[0]->x[0] + (e->v[0]->x[1]-e->next_active->v[0]->x[1] +
         e->next_active->m*(e->next_active->v[0]->x[0]-e->v[0]->x[0]))/
              (e->next_active->m - e->m);
    if ( (u > e->v[0]->x[0]-1e-10) && (u < ev.time) && (u < e->v[1]->x[0] - 1e-10)
             && (u < e->next_active->v[1]->x[0] - 1e-10) )
    { ev.e1 = e;
      ev.e2 = e->next_active; 
      ev.time = u;
    }
  }
 
  /* Below */
  if ( e->prev_active && (e->m < e->prev_active->m) && (e->v[1] != e->prev_active->v[1]))
  { REAL u;
    u = e->v[0]->x[0] + (e->v[0]->x[1]-e->prev_active->v[0]->x[1] +
         e->prev_active->m*(e->prev_active->v[0]->x[0]-e->v[0]->x[0]))/
              (e->prev_active->m - e->m);
    if ( (u > e->v[0]->x[0]-1e-8) && (u < ev.time) && (u < e->v[1]->x[0] - 1e-10)
             && (u < e->prev_active->v[1]->x[0] - 1e-10) ) 
    { ev.e1 = e->prev_active;
      ev.e2 = e; 
      ev.time = u;
    }
  }

  if ( ev.time < 1e20 )
  { 
    if ( visdebuglevel >= VIS_EVENTDUMP )
      printf("next crossing %d %d at %20.15f\n",
        (int)(ev.e1-vis_conedges),(int)(ev.e2-vis_conedges), (DOUBLE)ev.time);
    vis_insert_heap(&ev);
  }

} // end find_next_event()

/* for debugging */
/**************************************************************************
*
* function: dump_vislist()
*
* purpose: dump edge list in case of error.
*/

void dump_vislist()
{ struct vis_conedge *e;
  REAL v;
  int i;

  fprintf(stderr,"Visibility edge list debug dump at u = %18.15f; debug_seq %d\n",
      sweep_u,debug_seq);
  for ( e = active_edge_first ; e != &sentinel ; e = e->next_active )
  { if ( e == NULL ) { fprintf(stderr,"NULL next_active.\n"); break; }
    v = e->m*(sweep_u-e->v[0]->x[0]) + e->v[0]->x[1]; 
    fprintf(stderr,"%3d   %5d v: %18.15f layers:",e->seqno,e->id,v);
    for ( i = 0 ; i < e->layers ; i++ ) 
      fprintf(stderr," %3d",ordinal(e->layer[i]->f_id)+1);
    fprintf(stderr,"\n");
 }
 fprintf(stderr,"\n");
}

void check_vislist(REAL u)  /* for v in ascending sequence */
{ struct vis_conedge *e;
  REAL v,prev = -1e30;

  for ( e = active_edge_first ; e != &sentinel ; e = e->next_active )
  { if ( e == NULL ) { fprintf(stderr,"NULL next_active.\n"); break; }
    v = e->m*(u-e->v[0]->x[0]) + e->v[0]->x[1]; 
    if ( prev > v+1e-12 )
    {  dump_vislist();
       printf("Bad vislist at debug_seq %d\n",debug_seq);
    }
    prev = v;
 }
}



/***************************************************************************
*
* function: activate_edge()
*
* purpose: add edge to active edge list, if it is not there.
*          Returns insertion coordinate for debugging.
*          Does linear search, so could be made more efficient.
*
*/
REAL vis_eps = 1e-13; /* for equality detection */

REAL activate_edge(struct vis_conedge *e)
{ struct vis_conedge *spot;
  REAL v,vprev;
  int seq = 0;

  v = e->m*(sweep_u-e->v[0]->x[0]) + e->v[0]->x[1]; 
  if ( e->next_active ) return v; /* already active */

  vprev = -1e30;
  for ( spot = active_edge_first ; spot != NULL ; spot = spot->next_active )
  { REAL spotv = spot->m*(sweep_u-spot->v[0]->x[0]) + spot->v[0]->x[1];
    if ( spotv < vprev-1e-10 )
    { sprintf(errmsg,"Internal error: visibility list out of order by %f.\n",
        (DOUBLE)(vprev-spotv));
      kb_error(3509,errmsg,WARNING);
    }
    vprev = spotv; /* debugging */
    spot->seqno = seq++;
    if ( spotv > v+vis_eps ) 
      break;
    if ( (spotv > v-vis_eps) && (spot->m > e->m) ) 
       break;
  } 
  vprev = v;
  /* Have now located insertion spot; "spot" comes after e */
  if ( active_edge_first == spot ) active_edge_first = e;
  if ( spot->prev_active ) spot->prev_active->next_active = e;
  e->prev_active = spot->prev_active;
  spot->prev_active = e;
  e->next_active = spot;
  if ( e->prev_active )
  { int i;
    e->layer = (struct tsort**)temp_calloc(
       e->prev_active->layers+4, sizeof(struct tsort*));
    e->maxlayers = e->prev_active->layers+4;
    for ( i = 0 ; i < e->prev_active->layers ; i++ )
    {
      add_layer(e,e->prev_active->layer[i]);
    }
  }
  else
  { e->layer = (struct tsort**)temp_calloc(10, sizeof(struct tsort*));
    e->maxlayers = 10;
  }

  return v;
} // end activate_edge()

/*****************************************************************************
*
* function: deactivate_edge()
*
* purpose: 
*/
void deactivate_edge(struct vis_conedge *e)
{
  if ( e->prev_active )
    e->prev_active->next_active = e->next_active;
  else active_edge_first = e->next_active;
  e->next_active->prev_active = e->prev_active;
  e->next_active = e->prev_active = NULL;
  temp_free((char*)e->layer);
  e->layer = NULL;
  e->layers = 0;

} // end check_deactivate()

/*************************************************************************************
* 
* function: vis_edge_comp()
*
* purpose: evaluate active edge order.
*
*/

int vis_edge_comp(struct vis_conedge *a, struct vis_conedge *b, REAL u)
{ REAL va,vb;

  if ( a->v[0] == b->v[0] ) 
  { // compare slopes
    if ( a->m < b->m ) return -1;
    if ( a->m > b->m ) return  1;
    return 0;
  }
  if ( a->v[1] == b->v[1] ) 
  { // compare slopes
    if ( a->m < b->m ) return  1;
    if ( a->m > b->m ) return -1;
    return 0;
  }
  // now compare vertical position
  va = a->v[0]->x[1] + a->m*(u-a->v[0]->x[0]);
  vb = b->v[0]->x[1] + b->m*(u-b->v[0]->x[0]);
  if ( va < vb ) return -1;
  if ( va > vb ) return  1;

  // Geometric tie, so some arbitrary ordering
  if ( a < b ) return -1;
  if ( a > b ) return  1;
  return 0;
}


/*************************************************************************************
*
* function: visibility_recover()
*
* purpose: re-construct edge list, layers, and crossing-event list when 
*          things go wrong.
*
* return: 1 for success, -1 for failure
*/
int visibility_recover()
{ struct vis_conedge *e,*prev_e,*e_next;
  int j;
  REAL uv = vis_vertices[vertex_event_spot+1].x[0];  // next vertex after this one
  REAL ux = vis_heap[0].time;  // next crossing time
  REAL u = (sweep_u + (uv < ux ? uv : ux))/2;

  // Make sure edge list in order, using insertion sort
  for ( e = active_edge_first->next_active ; e ; e = e_next )
  { e_next = e->next_active;

    prev_e = e->prev_active;
    while ( vis_edge_comp(e,prev_e,u) < 0 )
    { 
      prev_e = prev_e->prev_active;
      if ( prev_e == NULL ) 
        break;
    }
    if ( prev_e == e->prev_active )
        continue;  // all ok

    // excise from current spot
    e->next_active->prev_active = e->prev_active;
    e->prev_active->next_active = e->next_active;

    // insert in new spot
    e->prev_active = prev_e;
    if ( prev_e )
    { e->next_active = prev_e->next_active;
      prev_e->next_active->prev_active = e;
      prev_e->next_active = e;
    }
    else
    { e->next_active = active_edge_first;
      active_edge_first->prev_active = e;
      active_edge_first = e;
    }
  } // end active list ordering

  // Now layers
  // First, clear old stuff.
  for ( e = active_edge_first ; e != NULL ; e = e->next_active )
  { e->layers = 0;
    memset(e->layer,0,e->maxlayers*sizeof(struct tsort*));
  }
  // Then restore layers
  for ( e = active_edge_first ; e != NULL ; e = e->next_active )
  { for ( j = e->rawstart ; j <= e->rawend ; j++ )
     { struct vis_rawedge *re = rawplist[j];
       if ( re->flags & V_FACET_BOTTOM )
       { struct vis_conedge *ee;
         // install going up until reach top edge of facet
         add_layer(e,re->tsrt);
         // and go up until hit top edge of facet
         ee = e;
         while ( ee )
         { int m;
           ee = ee->next_active;
           if ( ee == NULL ) break;
           for ( m = ee->rawstart ; m <= ee->rawend ; m++ )
             if ( rawplist[m]->tsrt == re->tsrt )
             { // have top, so stop adding as layer
               goto layer_end;
             }
           add_layer(ee,re->tsrt);
         }
         if ( ee == NULL )
         { int n,m;

           kb_error(3038,"visibility_test INTERNAL ERROR: can't find facet top edge.\n",WARNING);

           for ( ee = active_edge_first, n = 0 ; ee ; ee = ee->next_active, n++ )
           { for ( m = ee->rawstart ; m <= ee->rawend ; m++ )
               if ( rawplist[m]->tsrt == re->tsrt )
               {  printf ("tsrt at spot %d  flag %d\n",n,rawplist[m]->flags);
               }
                
            }
            return -1;
         }
       }
       layer_end: continue;
    }
  } // end layer restore

  // Rebuild crossing event heap
  vis_heap_count = 0;
  for ( e = active_edge_first ; e != &sentinel ; e = e->next_active )
    find_next_event(e);

  // Mark visible facets
  for ( e = active_edge_first ; e != &sentinel ; e = e->next_active )
    check_one_visible(e);

  return 1;

} // end visibility_recover

/***************************************************************************************
*
* function: handle_vertex_event()
*
* purpose: Handle the next vertex event. Do deactivation of left edges and activation
*          of right edges;
*
* return: 1 for success, -1 for error
*/
int handle_vertex_event(struct vis_vertex *v)
{  struct vis_conedge *before; // active edge below our vertex
   struct vis_conedge *after;  // active edge above our vertex
   struct vis_conedge *spot,*ee,*e,*last_e;
   int i,j,m;
   int retval = 1;  // exit code

   static int this_seq = 0;  // for detecting loops in linked list

   if ( visdebuglevel >= VIS_EVENTDUMP )
       printf("Vertex event at u = %18.15f  v = %18.15f\n",sweep_u,v->x[1]);

   // First, delete left edges and associated facets.
   if ( v->left_edge_head )
   {  // left edges in vislist might be out of order, if coincident
      // so delete them individually
      for ( spot = v->left_edge_head ; spot != NULL ; 
             spot = spot->left_edge_next )
      { 
        // Delete facets
        for ( j = spot->rawstart ; j <= spot->rawend ; j++ )
        { struct vis_rawedge *re = rawplist[j];
          struct vis_conedge *ee = spot->next_active;
          if ( re->flags & V_FACET_BOTTOM )
          { re->tsrt->flag &= ~VISIBILITY_LIVE;
            while ( ee )
            { if ( !delete_layer(ee,re->tsrt) )
                break;
              ee = ee->next_active;
            }
          }
        }

        // save insertion spot when get to last deletion
        if ( spot->left_edge_next == NULL )
        { before = spot->prev_active;
          after = spot->next_active;
        }

        // delete from linked list of edges
        deactivate_edge(spot);        
      }
   }
   else // have to do linear search through active edges
   { this_seq++;
     for ( spot = active_edge_first ; spot ; spot = spot->next_active )
     { REAL spoty = spot->m*(sweep_u-spot->v[0]->x[0]) + spot->v[0]->x[1];
       if ( spoty > v->x[1] )
           break;
       if ( spot->seqno == this_seq )
       {  kb_error(6542,"visibility_test INTERNAL ERROR: loop in edge list\n",WARNING);
          return -1;
       }
       spot->seqno = this_seq;     
     }
     before = spot->prev_active;
     after  = spot;
   }

   // Now insert right edges in place of left edges
   if ( v->right_edge_head )
   { if ( before )
       before->next_active = v->right_edge_head;
     else
       active_edge_first = v->right_edge_head;
     v->right_edge_head->prev_active = before;
     for ( ee = v->right_edge_head ; ee->right_edge_next != NULL ;
                 ee = ee->right_edge_next )
     {  ee->next_active = ee->right_edge_next;
        ee->right_edge_next->prev_active = ee;
     }
     ee->next_active = after;
     after->prev_active = ee;
   }
   else
   { if ( before )
       before->next_active = after;
     else
       active_edge_first = after;
     after->prev_active = before;
   }

   if ( v->right_edge_head == NULL )
     return 1;

   /* Fix up layers */
   // Add own facets of right edges
   for ( e = v->right_edge_head ; e != NULL ; e = e->right_edge_next )
   {
     for ( j = e->rawstart ; j <= e->rawend ; j++ )
     { struct vis_rawedge *re = rawplist[j];
       if ( re->flags & V_FACET_BOTTOM )
       { struct vis_conedge *ee;
         // install going up until reach top edge of facet
         add_layer(e,re->tsrt);
         // and go up until hit top edge of facet
         ee = e;
         while ( ee )
         { ee = ee->next_active;
           if ( ee == &sentinel ) break;
           for ( m = ee->rawstart ; m <= ee->rawend ; m++ )
             if ( rawplist[m]->tsrt == re->tsrt )
             { // have top, so stop adding as layer
               goto own_layer_end;
             }
           add_layer(ee,re->tsrt);
         }
         if ( ee == &sentinel )
         {
            retval = -1;
         }
       }
own_layer_end: continue;
     }
     if ( e->right_edge_next == NULL )
       last_e = e;
   }
   // Add continuing layers from below
   if ( before )
   { for ( i = 0 ; i < before->layers ; i++ )
     { for ( ee = v->right_edge_head ; ee != NULL ; ee = ee->right_edge_next )
       { // first see if this is the top of the facet
         for ( m = ee->rawstart ; m <= ee->rawend ; m++ )
           if ( rawplist[m]->tsrt == before->layer[i] )
           { // have top, so stop adding as layer
             goto prev_layer_end;
           }
         add_layer(ee,before->layer[i]);
       }
prev_layer_end: continue;
     }
   }

  // Look for upcoming crossing events
  find_next_event(v->right_edge_head);
  if ( last_e != v->right_edge_head )
    find_next_event(last_e);

  return retval;

}  // end handle_vertex_event


/***************************************************************************
*
* function: vis_crossing()
*
* purpose: Handle crossing of two edges.  Have to check to be sure 
*          switching order is needed, since crossing events can be
*          listed twice.
*
*/

void vis_crossing(
  struct vis_conedge *ea,
  struct vis_conedge *eb  /* ea below eb */
    )
{ struct vis_conedge *te;
  struct vis_rawedge *ra,*rb;
  int i;

  if ( ea->next_active != eb ) goto vis_crossing_exit; /* not adjacent */
  if ( ea != eb->prev_active )
     kb_error(2547,"Inconsistent active list.\n",RECOVERABLE);
  vis_crossing_count++;

  /* fix up layer lists, using info back in raw edge list */
  for ( i = ea->rawstart; i <= ea->rawend ; i++ )
  { ra = rawplist[i];
    if ( ra->flags & V_FACET_BOTTOM )
      delete_layer(eb,ra->tsrt);
    if ( ra->flags & V_FACET_TOP )
      add_layer(eb,ra->tsrt);
  }
  for ( i = eb->rawstart; i <= eb->rawend ; i++ )
  { rb = rawplist[i];
    if ( rb->flags & V_FACET_BOTTOM )
      add_layer(ea,rb->tsrt);
    if ( rb->flags & V_FACET_TOP )
      delete_layer(ea,rb->tsrt);
  }

  /* switch order in active list */
  eb->next_active->prev_active = ea;
  if ( ea->prev_active ) ea->prev_active->next_active = eb;
  else active_edge_first = eb;
  te = ea->prev_active;
  ea->prev_active = eb;
  eb->prev_active = te;
  te = eb->next_active;
  eb->next_active = ea;
  ea->next_active = te;

  /* Test for next events */ 
  find_next_event(ea);
  find_next_event(eb);
  
vis_crossing_exit: ;

}  // end vis_crossing()


/************************************************************************
*
* function: check_one_visible()
*
* purpose: mark top layer facet at given edge
*/
void check_one_visible(struct vis_conedge *e)
{ struct tsort *topf;
  int j;

  // using tsort order from painter's algorithm to see which is on top
  topf = NULL;
  for ( j = 0 ; j < e->layers ; j++ )
  { 
    struct tsort *f = e->layer[j];
    if ( (topf==NULL || (f > topf)) && (f->flag & VISIBILITY_LIVE) )    /* using painter facet order */
    { topf = f; }
  }
  if ( topf ) 
  { topf->flag |= VISIBLE;
    if ( visdebuglevel >= VIS_EVENTDUMP )
        printf("Marking facet %d visible at edge %d.\n",(int)(topf->f_id & 0xFFFFF)+1,(int)(e-vis_conedges));
  }
  e->flags &= ~V_LAYER_CHECK;
}

/*************************************************************************
*
* function: check_visible()
*
* purpose: Inspect layers of changed edge stacks and mark topmost facets.
*          Uses facet order from painter algorithm rather than
*          calculating local z, since that is subject to problems with
*          abutting facets. 
*
*/

void check_visible()
{
  int i;

  for ( i = 0 ; i < check_list_count ; i++ )
  { struct vis_conedge *e = check_list[i];

    if ( e->layers )
      check_one_visible(e);
  }
  check_list_count = 0;

} // end check_visible()


