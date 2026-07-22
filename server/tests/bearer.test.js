const { getBearerToken } = require('../src/middleware/bearer');

describe('getBearerToken', () => {
  const withHeader = (authorization) => ({ headers: { authorization } });

  it('returns the token from a well-formed Bearer header', () => {
    expect(getBearerToken(withHeader('Bearer abc.def.ghi'))).toBe('abc.def.ghi');
  });

  it('trims surrounding whitespace', () => {
    expect(getBearerToken(withHeader('Bearer   abc  '))).toBe('abc');
  });

  it('returns null when the header is missing', () => {
    expect(getBearerToken({ headers: {} })).toBeNull();
  });

  it('returns null when the scheme is not Bearer', () => {
    expect(getBearerToken(withHeader('Basic abc'))).toBeNull();
  });

  it('returns null when the token is empty', () => {
    expect(getBearerToken(withHeader('Bearer '))).toBeNull();
    expect(getBearerToken(withHeader('Bearer    '))).toBeNull();
  });
});
