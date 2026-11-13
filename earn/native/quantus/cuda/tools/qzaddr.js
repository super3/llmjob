#!/usr/bin/env node
'use strict';

// Prints a well-formed Quantus address for a random 32-byte account that has no
// key behind it: only for logging a miner in to tools/mockpool.js, which pays
// nothing and talks to no network. Never mine to it on a real pool.
const crypto = require('crypto');
const path = require('path');
const { encodeQuantusAddress } = require(path.join(__dirname, '..', '..', '..', '..', 'src', 'shared', 'quantus', 'address'));

console.log(encodeQuantusAddress(crypto.randomBytes(32)));
