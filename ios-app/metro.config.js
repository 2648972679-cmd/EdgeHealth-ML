const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Add .onnx as an asset extension so Metro bundles the model file
config.resolver.assetExts.push('onnx');

module.exports = config;
