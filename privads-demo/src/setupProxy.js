const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Proxy ONNX model requests to serve with correct MIME type
  app.use(
    '/models/interaction_predictor.onnx',
    createProxyMiddleware({
      target: 'http://localhost:3000',
      changeOrigin: true,
      pathRewrite: {
        '^/models/interaction_predictor.onnx': '/models/interaction_predictor.onnx'
      },
      onProxyReq: (proxyReq, req, res) => {
        // Set correct MIME type for ONNX files
        res.setHeader('Content-Type', 'application/wasm');
      }
    })
  );
};