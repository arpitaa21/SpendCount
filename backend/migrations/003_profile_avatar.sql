-- Optional profile image stored as a resized data URL. Existing profile records remain valid.
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_data_url TEXT;
