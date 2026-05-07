/*
Minimal inference for exported scikit-learn Pipeline (StandardScaler + MLPRegressor).

Usage in browser:
  <script src="xg_mlp_infer.js"></script>
  <script>
    loadXgModel('../models/xg_mlp_web.json').then(({predict}) => {
      const xg = predict(102, 40);
      console.log('xG:', xg.toFixed(4));
    });
  </script>
*/

(function(global){
  function relu(v){ return v > 0 ? v : 0; }

  function dotVecMat(vec, mat){
    // vec: [in], mat: [in][out] (same as numpy (1,in) @ (in,out))
    const out = new Array(mat[0].length).fill(0);
    for (let j = 0; j < mat[0].length; j++){
      let s = 0;
      for (let i = 0; i < vec.length; i++) s += vec[i] * mat[i][j];
      out[j] = s;
    }
    return out;
  }

  function addBias(vec, bias){
    const out = new Array(vec.length);
    for (let i = 0; i < vec.length; i++) out[i] = vec[i] + bias[i];
    return out;
  }

  function applyActivation(vec, name){
    if (!name || name === 'identity') return vec.slice();
    if (name === 'relu') return vec.map(relu);
    throw new Error('Unsupported activation: ' + name);
  }

  function makePredictor(model){
    const mean = model.scaler.mean;
    const scale = model.scaler.scale;
    const coefs = model.mlp.coefs;
    const intercepts = model.mlp.intercepts;
    const hiddenAct = model.mlp.activation || 'relu';
    const outAct = model.mlp.out_activation || 'identity';
    const clip = model.clip || [0,1];

    function transformXY(x, y){
      // StandardScaler: (x - mean) / scale
      return [ (x - mean[0]) / scale[0], (y - mean[1]) / scale[1] ];
    }

    function fwd(x, y){
      let z = transformXY(x, y);
      for (let li = 0; li < coefs.length; li++){
        z = dotVecMat(z, coefs[li]);
        z = addBias(z, intercepts[li]);
        if (li < coefs.length - 1){
          z = applyActivation(z, hiddenAct);
        }
      }
      // Output activation
      z = applyActivation(z, outAct);
      let v = z[0];
      if (Number.isFinite(clip[0]) && Number.isFinite(clip[1])){
        if (v < clip[0]) v = clip[0];
        if (v > clip[1]) v = clip[1];
      }
      return v;
    }

    return { predict: fwd };
  }

  async function loadXgModel(url){
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch model JSON: ' + res.status);
    const model = await res.json();
    return makePredictor(model);
  }

  // UMD style export
  global.loadXgModel = loadXgModel;
  global.__makeXgPredictor = makePredictor; // for testing if model JSON already in memory
})(typeof window !== 'undefined' ? window : globalThis);
