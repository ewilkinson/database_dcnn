create extension plpythonu;


CREATE FUNCTION distance2 (a float8[], b float8[] )
  RETURNS float8
AS $$
  import numpy as np
  c = np.sum(np.square(np.asarray(a)-np.asarray(b)))
  return c
$$ LANGUAGE plpythonu;


--select distance2(t1.data, test.data) from t1, test;
