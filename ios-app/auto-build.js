/**
 * Auto-build script: spawns EAS CLI and answers ALL interactive prompts.
 * This is needed because the first iOS build requires interactive credential setup.
 */
const { spawn } = require('child_process');

const ANSWERS = [
  'yes',  // Generate new certificate?
  'yes',  // Would you like to create a new one?
  'yes',  // Generate new provisioning profile?
  'yes',  // Various confirmations
  'yes',
  'yes',
  'yes',
  'yes',
  'yes',
];

let answerIndex = 0;

const child = spawn('npx', ['eas-cli', 'build', '--platform', 'ios', '--profile', 'production'], {
  env: {
    ...process.env,
    EXPO_TOKEN: '1tBjSz8GG6u5rq3penh3Y90YWKd_XDwLJkuC0Kww',
    EXPO_APPLE_ID: '2648972679@qq.com',
    EXPO_APPLE_APP_SPECIFIC_PASSWORD: 'aqwk-lztu-wxrl-zmnd',
    EAS_BUILD_NO_EXPO_GO_WARNING: 'true',
    CI: 'true',
  },
  stdio: ['pipe', 'pipe', 'pipe'],
  shell: true,
});

child.stdout.on('data', (data) => {
  const text = data.toString();
  process.stdout.write(text);

  // Auto-answer prompts
  const lower = text.toLowerCase();
  if (
    lower.includes('yes/no') ||
    lower.includes('(y/n)') ||
    lower.includes('generate a new') ||
    lower.includes('create a new') ||
    lower.includes('would you like') ||
    lower.includes('do you want') ||
    lower.includes('proceed') ||
    lower.includes('continue')
  ) {
    if (answerIndex < ANSWERS.length) {
      const answer = ANSWERS[answerIndex++];
      console.log(`>>> Auto-answering: ${answer}`);
      child.stdin.write(answer + '\n');
    }
  }

  // Auto-answer Apple ID questions
  if (lower.includes('apple id') || lower.includes('username')) {
    console.log('>>> Auto-answering: Apple ID');
    child.stdin.write('2648972679@qq.com\n');
  }
});

child.stderr.on('data', (data) => {
  process.stderr.write(data);
});

child.on('close', (code) => {
  console.log(`Build process exited with code ${code}`);
  process.exit(code);
});

// Keep stdin alive
process.stdin.resume();
