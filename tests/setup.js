// Test setup file
jest.setTimeout(10000);

// Mock environment variables
process.env.NODE_ENV = 'test';
process.env.CLERK_PUBLISHABLE_KEY = 'test_publishable_key';
process.env.CLERK_SECRET_KEY = 'test_secret_key';