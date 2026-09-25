// Signups for the managed-mining service. Public and unauthenticated (the form
// on /managed posts here), so everything is validated and clamped at the
// boundary, and a repeat signup updates the existing row instead of adding one.

const MAX_EMAIL = 254;   // RFC 5321 path limit
const MAX_NOTE = 1000;
const MAX_SOURCE = 64;
const MAX_GPUS = 100000;

// Deliberately loose: one @, something on both sides, a dot in the domain, no
// whitespace. Anything stricter rejects real addresses; deliverability is only
// ever proven by actually emailing it.
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function normalizeEmail(email) {
  return String(email == null ? '' : email).trim().toLowerCase();
}

function isValidEmail(email) {
  return email.length > 0 && email.length <= MAX_EMAIL && EMAIL_RE.test(email);
}

// GPU count as a whole number in [0, MAX_GPUS], or null when not given / junk.
function parseGpus(v) {
  if (v == null || v === '') return null;
  const n = Math.floor(Number(v));
  if (!Number.isFinite(n) || n < 0) return null;
  return Math.min(n, MAX_GPUS);
}

function clampText(v, max) {
  const s = String(v == null ? '' : v).trim();
  return s ? s.slice(0, max) : null;
}

class WaitlistService {
  constructor(db) {
    this.db = db;
  }

  async join(input = {}) {
    const email = normalizeEmail(input.email);
    if (!isValidEmail(email)) return { error: 'Please enter a valid email address' };

    const gpus = parseGpus(input.gpus);
    const note = clampText(input.note, MAX_NOTE);
    const source = clampText(input.source, MAX_SOURCE);
    const now = Date.now();

    // COALESCE keeps an earlier answer when a repeat signup leaves a field blank.
    await this.db.query(
      `INSERT INTO managed_waitlist (email, gpus, note, source, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $5)
       ON CONFLICT (email) DO UPDATE SET
         gpus = COALESCE(EXCLUDED.gpus, managed_waitlist.gpus),
         note = COALESCE(EXCLUDED.note, managed_waitlist.note),
         source = COALESCE(EXCLUDED.source, managed_waitlist.source),
         updated_at = EXCLUDED.updated_at`,
      [email, gpus, note, source, now]
    );
    return { success: true };
  }
}

WaitlistService.normalizeEmail = normalizeEmail;
WaitlistService.isValidEmail = isValidEmail;
WaitlistService.parseGpus = parseGpus;
WaitlistService.clampText = clampText;

module.exports = WaitlistService;
