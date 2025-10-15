Isolated Maximum-Weight Matching (MWM) and visualization backend developed for quantum error-correction (QEC) research.  

# Matching Backend

This project uses **Joris van Rantwijk’s pure-Python maximum-weight matching** implementation (mwmatching.py) for general graphs.

- **Algorithm:** Galil’s \(O(n^3)\) maximum-weight matching algorithm for general graphs  
- **Source:** Joris van Rantwijk, *Maximum Weighted Matching*  
- **Original file:** `mwmatching.py`

## Reference

Joris van Rantwijk, *Maximum Weighted Matching* — [https://jorisvr.nl/article/maximum-matching](https://jorisvr.nl/article/maximum-matching)

## Notes

- Works well for small to medium graph sizes  
- Pure-Python and easy to modify  
- Performance is interpreter-bound; author notes ~10× speedup possible if compiled
