import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

const errors = new Counter('errors');
const successRate = new Rate('success_rate');
const streamDuration = new Trend('stream_duration_ms');

export let options = {
  stages: [
    { duration: '10s', target: 10 },   // Ramp up to 10 users
    { duration: '30s', target: 50 },   // Ramp up to 50 users
    { duration: '60s', target: 50 },   // Hold at 50 concurrent
    { duration: '10s', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'],  // 95% of requests complete within 5s
    http_req_failed: ['rate<0.10'],     // Less than 10% failure rate
    success_rate: ['rate>0.90'],        // More than 90% success
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8200';
const TEST_TOKEN = __ENV.TEST_TOKEN || '';

const TEST_MESSAGES = [
  'Summarize this meeting: Team discussed Q2 roadmap. Alice will lead mobile feature. Bob handles backend. Deadline end of June.',
  'What are the key action items from: Project review showed 3 bugs in production. Fix authentication by Friday. Improve logging by next week.',
  'Analyze the discussion: Sales team reported 20% growth. Marketing needs more budget. Engineering requested 2 new hires.',
  'Extract tasks from: Design team will deliver mockups Monday. Development starts Tuesday. Testing phase is Wednesday to Friday.',
];

export default function () {
  const message = TEST_MESSAGES[Math.floor(Math.random() * TEST_MESSAGES.length)];
  const start = Date.now();

  const response = http.post(
    `${BASE_URL}/chat`,
    JSON.stringify({ message }),
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${TEST_TOKEN}`,
        'Accept': 'text/event-stream',
      },
      timeout: '10s',
    }
  );

  const duration = Date.now() - start;
  streamDuration.add(duration);

  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'response has content': (r) => r.body && r.body.length > 0,
    'no server error': (r) => r.status < 500,
  });

  if (!success) {
    errors.add(1);
  }
  successRate.add(success);

  sleep(0.5);
}

export function handleSummary(data) {
  return {
    'tests/load/results/summary.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, opts) {
  const { metrics } = data;
  const lines = [
    '\n=== Load Test Summary ===',
    `VUs: ${metrics.vus ? metrics.vus.value : 'N/A'}`,
    `Requests: ${metrics.http_reqs ? metrics.http_reqs.count : 'N/A'}`,
    `Failed: ${metrics.http_req_failed ? (metrics.http_req_failed.rate * 100).toFixed(2) : 'N/A'}%`,
    `p95 Duration: ${metrics.http_req_duration ? metrics.http_req_duration['p(95)'].toFixed(0) : 'N/A'}ms`,
    '========================\n',
  ];
  return lines.join('\n');
}
