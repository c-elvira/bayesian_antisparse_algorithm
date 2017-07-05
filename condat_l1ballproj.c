/*
 #  File            : condat_l1ballproj.c
 #
 #  Version History : 1.0, Aug. 15, 2014
 #
 #  Author          : Laurent Condat, PhD, CNRS research fellow in France.
 #
 #  Description     : This file contains an implementation in the C language
 #                    of algorithms described in the research paper:
 #
 #                    L. Condat, "Fast Projection onto the Simplex and the
 #                    l1 Ball", preprint Hal-01056171, 2014.
 #
 #                    This implementation comes with no warranty: due to the
 #                    limited number of tests performed, there may remain
 #                    bugs. In case the functions would not do what they are
 #                    supposed to do, please email the author (contact info
 #                    to be found on the web).
 #
 #                    If you use this code or parts of it for any purpose,
 #                    the author asks you to cite the paper above or, in
 #                    that event, its published version. Please email him if
 #                    the proposed algorithms were useful for one of your
 #                    projects, or for any comment or suggestion.
 #
 #  Usage rights    : Copyright Laurent Condat.
 #                    This file is distributed under the terms of the CeCILL
 #                    licence (compatible with the GNU GPL), which can be
 #                    found at the URL "http://www.cecill.info".
 #
 #  This software is governed by the CeCILL license under French law and
 #  abiding by the rules of distribution of free software. You can  use,
 #  modify and or redistribute the software under the terms of the CeCILL
 #  license as circulated by CEA, CNRS and INRIA at the following URL :
 #  "http://www.cecill.info".
 #
 #  As a counterpart to the access to the source code and rights to copy,
 #  modify and redistribute granted by the license, users are provided only
 #  with a limited warranty  and the software's author,  the holder of the
 #  economic rights,  and the successive licensors  have only  limited
 #  liability.
 #
 #  In this respect, the user's attention is drawn to the risks associated
 #  with loading,  using,  modifying and/or developing or reproducing the
 #  software by the user in light of its specific status of free software,
 #  that may mean  that it is complicated to manipulate,  and  that  also
 #  therefore means  that it is reserved for developers  and  experienced
 #  professionals having in-depth computer knowledge. Users are therefore
 #  encouraged to load and test the software's suitability as regards their
 #  requirements in conditions enabling the security of their systems and/or
 #  data to be ensured and,  more generally, to use and operate it in the
 #  same conditions as regards security.
 #
 #  The fact that you are presently reading this means that you have had
 #  knowledge of the CeCILL license and that you accept its terms.
*/


/* This code was compiled using
gcc -march=native -O2 condat_l1ballproj.c -o main  -I/usr/local/include/
	-lm -lgsl  -L/usr/local/lib/
On my machine, gcc is actually a link to the compiler Apple LLVM version 5.1
(clang-503.0.40) */


/* The following functions are implemented:
l1ballproj_IBIS
l1ballproj_Condat (proposed algorithm)
These functions take the same parameters. They project the vector y onto
the closest vector x of same length (parameter N in the paper) with
sum_{n=0}^{N-1}|x[n]|<=a.
We must have length>=1.
For l1ballproj_Condat only:
we can have x==y (projection done in place). If x!=y, the arrays x and y must
not overlap, as x is used for temporary calculations before y is accessed.
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define datatype double /* type of the elements in y */


/* Proposed algorithm */
static void l1ballproj_Condat(datatype* y, datatype* x,
const unsigned int length, const double a) {
	if (a<=0.0) {
		if (a==0.0) memset(x,0,length*sizeof(datatype));
		return;
	}
	datatype*	aux = (x==y ? (datatype*)malloc(length*sizeof(datatype)) : x);
	int		auxlength=1;
	int		auxlengthold=-1;
	double	tau=(*aux=(*y>=0.0 ? *y : -*y))-a;
	int 	i=1;
	for (; i<length; i++)
		if (y[i]>0.0) {
			if (y[i]>tau) {
				if ((tau+=((aux[auxlength]=y[i])-tau)/(auxlength-auxlengthold))
				<=y[i]-a) {
					tau=y[i]-a;
					auxlengthold=auxlength-1;
				}
				auxlength++;
			}
		} else if (y[i]!=0.0) {
			if (-y[i]>tau) {
				if ((tau+=((aux[auxlength]=-y[i])-tau)/(auxlength-auxlengthold))
				<=aux[auxlength]-a) {
					tau=aux[auxlength]-a;
					auxlengthold=auxlength-1;
				}
				auxlength++;
			}
		}
	if (tau<=0) {	/* y is in the l1 ball => x=y */
		if (x!=y) memcpy(x,y,length*sizeof(datatype));
		else free(aux);
	} else {
		datatype*  aux0=aux;
		if (auxlengthold>=0) {
			auxlength-=++auxlengthold;
			aux+=auxlengthold;
			while (--auxlengthold>=0)
				if (aux0[auxlengthold]>tau)
					tau+=((*(--aux)=aux0[auxlengthold])-tau)/(++auxlength);
		}
		do {
			auxlengthold=auxlength-1;
			for (i=auxlength=0; i<=auxlengthold; i++)
				if (aux[i]>tau)
					aux[auxlength++]=aux[i];
				else
					tau+=(tau-aux[i])/(auxlengthold-i+auxlength);
		} while (auxlength<=auxlengthold);
		for (i=0; i<length; i++)
			x[i]=(y[i]-tau>0.0 ? y[i]-tau : (y[i]+tau<0.0 ? y[i]+tau : 0.0));
		if (x==y) free(aux0);
	}
}


/* function eplb of the package SLEP, implementing the algorithm IBIS of the paper
J. Liu and J. Ye. Efficient Euclidean Projections in Linear Time, ICML 2009.
Version without the initial guess lambda0 of lambda (tau in our notations) */
static void l1ballproj_IBIS(datatype* y, datatype* x,
const unsigned int length, const double a) {
    #define delta 1e-13
    int i, j, flag=0;
    int rho_1, rho_2, rho, rho_T, rho_S;
    int V_i_b, V_i_e, V_i;
    double lambda_1, lambda_2, lambda_T, lambda_S, lambda;
    double s_1, s_2, s, s_T, s_S, y_max, temp;
    double f_lambda_1, f_lambda_2, f_lambda, f_lambda_T, f_lambda_S;
    int iter_step=0;

    /* find the maximal absolute value in y
     * and copy the (absolute) values from y to x
     */

	/*if (a< 0){
		printf("\n a should be nonnegative!");
		return;
	}*/

    V_i=0;
    if (y[0] !=0){
        rho_1=1;
        s_1=x[V_i]=y_max=fabs(y[0]);
        V_i++;
    }
    else{
        rho_1=0;
        s_1=y_max=0;
    }

    for (i=1;i<length; i++){
        if (y[i]!=0){
            x[V_i]=fabs(y[i]); s_1+= x[V_i]; rho_1++;

            if (x[V_i] > y_max)
                y_max=x[V_i];
            V_i++;
        }
    }

    /* If ||y||_1 <= a, then y is the solution  */
    if (s_1 <= a){
        flag=1;        lambda=0;
        for(i=0;i<length;i++){
            x[i]=y[i];
        }
        /**root=lambda;
        *steps=iter_step;*/
        return;
    }

    lambda_1=0; lambda_2=y_max;
    f_lambda_1=s_1 -a;
    /*f_lambda_1=s_1-rho_1* lambda_1 -a;*/
    rho_2=0; s_2=0; f_lambda_2=-a;
    V_i_b=0; V_i_e=V_i-1;

    /*lambda=lambda0;
    if ( (lambda<lambda_2) && (lambda> lambda_1) ){
         i=V_i_b; j=V_i_e; rho=0; s=0;
        while (i <= j){
            while( (i <= V_i_e) && (x[i] <= lambda) ){
                i++;
            }
            while( (j>=V_i_b) && (x[j] > lambda) ){
                s+=x[j];
                j--;
            }
            if (i<j){
                s+=x[i];

                temp=x[i];  x[i]=x[j];  x[j]=temp;
                i++;  j--;
            }
		}

        rho=V_i_e-j;  rho+=rho_2;  s+=s_2;
		f_lambda=s-rho*lambda-a;

        if ( fabs(f_lambda)< delta ){
            flag=1;
		}

		if (f_lambda <0){
			lambda_2=lambda; s_2=s;	rho_2=rho; f_lambda_2=f_lambda;

			V_i_e=j;  V_i=V_i_e-V_i_b+1;
		}
		else{
			lambda_1=lambda; rho_1=rho;	s_1=s; f_lambda_1=f_lambda;

			V_i_b=i; V_i=V_i_e-V_i_b+1;
		}

		if (V_i==0){
			lambda=(s - a)/ rho;
			flag=1;
		}
    }*/

    while (!flag){
        iter_step++;

        /* compute lambda_T  */
        lambda_T=lambda_1 + f_lambda_1 /rho_1;
        if(rho_2 !=0){
            if (lambda_2 + f_lambda_2 /rho_2 >	lambda_T)
                lambda_T=lambda_2 + f_lambda_2 /rho_2;
        }

        /* compute lambda_S */
        lambda_S=lambda_2 - f_lambda_2 *(lambda_2-lambda_1)/(f_lambda_2-f_lambda_1);

        if (fabs(lambda_T-lambda_S) <= delta){
            lambda=lambda_T; flag=1;
            break;
        }

        /* set lambda as the middle point of lambda_T and lambda_S */
        lambda=(lambda_T+lambda_S)/2;

        s_T=s_S=s=0;
        rho_T=rho_S=rho=0;
        i=V_i_b; j=V_i_e;
        while (i <= j){
            while( (i <= V_i_e) && (x[i] <= lambda) ){
                if (x[i]> lambda_T){
                    s_T+=x[i]; rho_T++;
                }
                i++;
            }
            while( (j>=V_i_b) && (x[j] > lambda) ){
                if (x[j] > lambda_S){
                    s_S+=x[j]; rho_S++;
                }
                else{
                    s+=x[j];  rho++;
                }
                j--;
            }
            if (i<j){
                if (x[i] > lambda_S){
                    s_S+=x[i]; rho_S++;
                }
                else{
                    s+=x[i]; rho++;
                }

                if (x[j]> lambda_T){
                    s_T+=x[j]; rho_T++;
                }

                temp=x[i]; x[i]=x[j];  x[j]=temp;
                i++; j--;
            }
		}

        s_S+=s_2; rho_S+=rho_2;
        s+=s_S; rho+=rho_S;
        s_T+=s; rho_T+=rho;
        f_lambda_S=s_S-rho_S*lambda_S-a;
        f_lambda=s-rho*lambda-a;
        f_lambda_T=s_T-rho_T*lambda_T-a;

        /*printf("\n %d & %d  & %5.6f & %5.6f & %5.6f & %5.6f & %5.6f \\\\ \n \\hline ", iter_step, V_i, lambda_1, lambda_T, lambda, lambda_S, lambda_2);*/

        if ( fabs(f_lambda)< delta ){
            /*printf("\n lambda");*/
            flag=1;
            break;
        }
        if ( fabs(f_lambda_S)< delta ){
           /* printf("\n lambda_S");*/
            lambda=lambda_S; flag=1;
            break;
        }
        if ( fabs(f_lambda_T)< delta ){
           /* printf("\n lambda_T");*/
            lambda=lambda_T; flag=1;
            break;
        }

        /*
        printf("\n\n f_lambda_1=%5.6f, f_lambda_2=%5.6f, f_lambda=%5.6f",f_lambda_1,f_lambda_2, f_lambda);
        printf("\n lambda_1=%5.6f, lambda_2=%5.6f, lambda=%5.6f",lambda_1, lambda_2, lambda);
        printf("\n rho_1=%d, rho_2=%d, rho=%d ",rho_1, rho_2, rho);
         */

        if (f_lambda <0){
            lambda_2=lambda;  s_2=s;  rho_2=rho;
            f_lambda_2=f_lambda;

            lambda_1=lambda_T; s_1=s_T; rho_1=rho_T;
            f_lambda_1=f_lambda_T;

            V_i_e=j;  i=V_i_b;
            while (i <= j){
                while( (i <= V_i_e) && (x[i] <= lambda_T) ){
                    i++;
                }
                while( (j>=V_i_b) && (x[j] > lambda_T) ){
                    j--;
                }
                if (i<j){
                    x[j]=x[i];
                    i++;   j--;
                }
            }
            V_i_b=i; V_i=V_i_e-V_i_b+1;
        }
        else{
            lambda_1=lambda;  s_1=s; rho_1=rho;
            f_lambda_1=f_lambda;

            lambda_2=lambda_S; s_2=s_S; rho_2=rho_S;
            f_lambda_2=f_lambda_S;

            V_i_b=i;  j=V_i_e;
            while (i <= j){
                while( (i <= V_i_e) && (x[i] <= lambda_S) ){
                    i++;
                }
                while( (j>=V_i_b) && (x[j] > lambda_S) ){
                    j--;
                }
                if (i<j){
                    x[i]=x[j];
                    i++;   j--;
                }
            }
            V_i_e=j; V_i=V_i_e-V_i_b+1;
        }

        if (V_i==0){
            lambda=(s - a)/ rho; flag=1;
            /*printf("\n V_i=0, lambda=%5.6f",lambda);*/
            break;
        }
    }/* end of while */


    for(i=0;i<length;i++){
        if (y[i] > lambda)
            x[i]=y[i]-lambda;
        else
            if (y[i]< -lambda)
                x[i]=y[i]+lambda;
            else
                x[i]=0;
    }
    /**root=lambda;
    *steps=iter_step;*/
}


/* Example of code to compare the computation times
Result on my machine: my algorithm is 2.5x faster than IBIS */
int main() {
	datatype*	y;
	datatype*	x;
	double	val=0.0;
	double 	aux;
	double 	epsi;
    int 	i,j;
    int 	Nbrea;
    unsigned int length;
    clock_t	start, end;
    const gsl_rng_type *T;
  	gsl_rng *r;
  	gsl_rng_env_setup();
	gsl_rng_default_seed=32067;
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	double timemean=0.0, timevar=0.0, timedelta;
	const double a = 1.0;
	srand((unsigned int)pow(time(NULL)%100,3));
    double*	 timetable;

    length = 1000000;
	Nbrea = 100;
	y=(datatype*)malloc(length*sizeof(datatype));
	x=(datatype*)malloc(length*sizeof(datatype));
	timetable=(double*)malloc((Nbrea+1)*sizeof(double));
	for (j=0; j<=Nbrea; j++) {
		for (i=0; i<length; i++) {
	    	y[i]=gsl_ran_gaussian(r, 0.01);
	    }
	    y[(int)(rand() / (((double)RAND_MAX+1.0)/length))]+=a;
	    start=clock();
	    l1ballproj_Condat(y,x,length,a);
	    /*l1ballproj_IBIS(y,x,length,a); */
	    end=clock();
	    timetable[j]=(double)(end-start)/CLOCKS_PER_SEC;
	}
	/* we discard the first value, because the computation time is higher,
	probably because of cache operations */
	for (j=1; j<=Nbrea; j++) {
		timedelta=timetable[j]-timemean;
		timemean+=timedelta/j;
		timevar+=timedelta*(timetable[j]-timemean);
	}
	timevar=sqrt(timevar/Nbrea);
	printf("av. time: %e std dev: %e\n",timemean,timevar);
	free(x);
    free(y);
    return 0;
}




