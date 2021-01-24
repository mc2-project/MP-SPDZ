#ifndef _random
#define _random

#include "Tools/octetStream.h"
#include "Tools/sha1.h"
#include "Tools/aes.h"
#include "Tools/avx_memcpy.h"
#include "Networking/data.h"

#include <mpir.h>

#define USE_AES

#ifndef USE_AES
  #define PIPELINES   1
  #define SEED_SIZE   HASH_SIZE
  #define RAND_SIZE   HASH_SIZE
#else
#ifdef __AES__
  #define PIPELINES   8
#else
  #define PIPELINES   1
#endif
  #define SEED_SIZE   AES_BLK_SIZE
  #define RAND_SIZE   (PIPELINES * AES_BLK_SIZE)
#endif


/* This basically defines a randomness expander, if using
 * as a real PRG on an input stream you should first collapse
 * the input stream down to a SEED, say via CBC-MAC (under 0 key)
 * or via a hash
 */

// __attribute__ is needed to get the sse instructions to avoid
//  seg faulting.
     
class PRNG
{
   octet seed[SEED_SIZE]; 
   octet state[RAND_SIZE] __attribute__((aligned (16)));
   octet random[RAND_SIZE] __attribute__((aligned (16)));

   #ifdef USE_AES
#ifdef __AES__
     bool useC;
#else
     const static bool useC = true;
#endif

     // Two types of key schedule for the different implementations 
     // of AES
     uint  KeyScheduleC[44];
     octet KeySchedule[176]  __attribute__((aligned (16)));
   #endif

   int cnt;    // How many bytes of the current random value have been used

   void hash(); // Hashes state to random and sets cnt=0
   void next();

   public:

   PRNG();

   // For debugging
   void print_state() const;

   // Set seed from dev/random
   void ReSeed();

   // Set seed from array
   void SetSeed(unsigned char*);
   void SetSeed(PRNG& G);
   void InitSeed();
   
   double get_double();
   bool get_bit() { return get_uchar() & 1; }
   unsigned char get_uchar();
   unsigned int get_uint();
   void get_bigint(bigint& res, int n_bits, bool positive = true);
   void get(bigint& res, int n_bits, bool positive = true);
   void get(int& res, int n_bits, bool positive = true);
   void randomBnd(bigint& res, const bigint& B, bool positive=true);
   bigint randomBnd(const bigint& B, bool positive=true);
   void randomBnd(mp_limb_t* res, const mp_limb_t* B, size_t n_bytes, mp_limb_t mask = -1);
   word get_word()
     {
       word a;
       get_octets<sizeof(a)>((octet*)&a);
       return le64toh(a);
     }
   __m128i get_doubleword();
   void get_octetStream(octetStream& ans,int len);
   void get_octets(octet* ans, int len);
   template <int L>
   void get_octets(octet* ans);

   const octet* get_seed() const
     { return seed; }

   template<class T>
   T get()
     { T res; res.randomize(*this); return res; }
};

class SeededPRNG : public PRNG
{
public:
  SeededPRNG()
  {
    ReSeed();
  }
};


inline unsigned char PRNG::get_uchar()
{
  if (cnt>=RAND_SIZE) { next(); }
  unsigned char ans=random[cnt];
  cnt++;
  // print_state(); cout << " UCHA " << (int) ans << endl;
  return ans;
}


inline __m128i PRNG::get_doubleword()
{
    if (cnt > RAND_SIZE - 16)
        next();
    __m128i ans = _mm_loadu_si128((__m128i*)&random[cnt]);
    cnt += 16;
    return ans;
}


inline void PRNG::get_octets(octet* ans,int len)
{
  int pos=0;
  while (len)
    {
      int step=min(len,RAND_SIZE-cnt);
      memcpy(ans+pos,random+cnt,step);
      pos+=step;
      len-=step;
      cnt+=step;
      if (cnt==RAND_SIZE)
        next();
    }
}

template<int L>
inline void PRNG::get_octets(octet* ans)
{
   if (L < RAND_SIZE - cnt)
   {
     avx_memcpy(ans, random + cnt, L);
     cnt += L;
   }
   else
     get_octets(ans, L);
}

#endif
