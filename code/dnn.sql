create extension plpythonu;


-- CREATE FUNCTION distance2 (a float8[], b float8[] )
--  RETURNS float8
-- AS $$
--  import numpy as np
--  c = np.sum(np.square(np.asarray(a)-np.asarray(b)))
--  return c
-- $$ LANGUAGE plpythonu;

CREATE FUNCTION distance2 (a float8[], b float8[] )
  RETURNS float8
AS $$
  import numpy as np
  A = np.asarray(a)
  B = np.asarray(b)

  weights_ij = np.ones(A.shape)

  sparse_j = A < -0.1
  sparse_i = B < -0.1

  weights_ij[sparse_i & sparse_j] = 1e-7

  c = np.sum( weights_ij * np.square(A - B)) / (np.sum(weights_ij))
  return c
$$ LANGUAGE plpythonu;

-- Hash index creation for quick lookup based on file
-- CREATE INDEX file_idx on pca_fc7 USING hash (file);

--select distance2(t1.data, test.data) from t1, test;
