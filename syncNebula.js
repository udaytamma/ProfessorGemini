#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-require-imports */

/**
 * syncNebula - Sync Nebula Documents to Cyrus
 *
 * Reads markdown files from source folders and generates TypeScript
 * data files for the Nebula sections in Cyrus.
 *
 * Can be run from ProfessorGemini directory:
 *   node syncNebula.js
 *
 * Sources:
 * - /Users/omega/Documents/Job Search/LLM Suggestions/ → Scratch Pad
 * - /Users/omega/Projects/Cyrus/gemini-responses/ → Knowledge Base
 *
 * See SCRIPTS.md for full documentation.
 */

const fs = require('fs');
const path = require('path');

// Absolute paths to Cyrus project
const CYRUS_ROOT = '/Users/omega/Projects/Cyrus';

// Configuration for each source
const SOURCES = {
  scratchPad: {
    name: 'Scratch Pad',
    sourceDir: '/Users/omega/Documents/Job Search/LLM Suggestions',
    outputFile: path.join(CYRUS_ROOT, 'src', 'data', 'scratch-pad.ts'),
    interfaceName: 'ScratchPadDoc',
    exportName: 'scratchPadDocs',
    getterName: 'getScratchPadDoc',
    slugsGetterName: 'getAllScratchPadSlugs',
    titleExtractor: extractTitleFromContent, // Complex extraction for LLM outputs
  },
  knowledgeBase: {
    name: 'Knowledge Base',
    sourceDir: path.join(CYRUS_ROOT, 'gemini-responses'),
    outputFile: path.join(CYRUS_ROOT, 'src', 'data', 'knowledge-base.ts'),
    interfaceName: 'KnowledgeBaseDoc',
    exportName: 'knowledgeBaseDocs',
    getterName: 'getKnowledgeBaseDoc',
    slugsGetterName: 'getAllKnowledgeBaseSlugs',
    titleExtractor: extractTitleFromFrontmatter, // Simple frontmatter extraction
  },
};

// Patterns that indicate a prompt rather than a proper title
const PROMPT_PATTERNS = [
  /^(okay|great|give me|suggest|help me|can you|please|i want|i need|here are my)/i,
  /\?$/,  // Ends with question mark
  /^[a-z]/,  // Starts with lowercase
];

// Maximum length for a "real" title
const MAX_TITLE_LENGTH = 80;

/**
 * Generate a URL-safe slug from a title
 */
function generateSlug(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '') // Remove special chars
    .replace(/\s+/g, '-')          // Spaces to hyphens
    .replace(/-+/g, '-')           // Collapse multiple hyphens
    .replace(/^-|-$/g, '')         // Trim leading/trailing hyphens
    .substring(0, 60);             // Limit length
}

/**
 * Check if a title looks like a prompt rather than a proper title
 */
function looksLikePrompt(title) {
  if (title.length > MAX_TITLE_LENGTH) return true;
  return PROMPT_PATTERNS.some(pattern => pattern.test(title.trim()));
}

/**
 * Extract a better title from content when H1 is a prompt
 */
function extractBetterTitle(content, h1Title) {
  // Strategy 1: Look for H1 inside a markdown code block (common in Perplexity outputs)
  const codeBlockH1Match = content.match(/```markdown\s*\n#\s+([^\n]+)/);
  if (codeBlockH1Match && !looksLikePrompt(codeBlockH1Match[1])) {
    return codeBlockH1Match[1].trim();
  }

  // Strategy 2: Map common prompt patterns to meaningful titles
  const promptMappings = [
    [/checklist/i, 'Work Session Guardrails'],
    [/here are my answers/i, 'Principal TPM Interview Prep Assessment'],
    [/system design.*suggest|suggest.*system design/i, 'System Design Practice Problems'],
    [/markdown file/i, 'LLMs and Principal TPM Work'],
  ];

  for (const [pattern, fallbackTitle] of promptMappings) {
    if (pattern.test(h1Title)) {
      return fallbackTitle;
    }
  }

  // Strategy 3: Look for bold text at the start that might be a title
  const boldMatch = content.match(/^\*\*([^*]+)\*\*/m);
  if (boldMatch && !looksLikePrompt(boldMatch[1]) && boldMatch[1].length < MAX_TITLE_LENGTH) {
    return boldMatch[1].trim();
  }

  // Strategy 4: Look for the first H2 that looks like a real title
  const h2Matches = content.match(/^##\s+(.+)$/gm);
  if (h2Matches) {
    for (const match of h2Matches) {
      const h2Title = match.replace(/^##\s+/, '').trim();
      if (!looksLikePrompt(h2Title) &&
          !h2Title.match(/^(what|where|how|why|when|summary|overview|introduction|background|\d+\.)/i) &&
          !h2Title.match(/^level \d/i)) {
        return h2Title;
      }
    }
  }

  // Strategy 5: Generate title from content analysis
  const topics = {
    'Interview Prep Plan': /interview|behavioral|practice|stories/gi,
    'System Design Practice': /system design|architecture|design exercise/gi,
    'TPM Checklist': /checklist|guardrails|rules|daily/gi,
    'Career Strategy': /career|strategy|assessment|gaps|strengths/gi,
    'LLM Usage Guidelines': /llm|ai|orchestrator|prompts/gi,
  };

  for (const [title, pattern] of Object.entries(topics)) {
    const matches = content.match(pattern);
    if (matches && matches.length >= 3) {
      return title;
    }
  }

  // Last resort: clean up the H1 and truncate
  return h1Title.substring(0, 60) + (h1Title.length > 60 ? '...' : '');
}

/**
 * Extract title from markdown content (for Scratch Pad - complex LLM outputs)
 */
function extractTitleFromContent(content) {
  // Look for first H1 header
  const h1Match = content.match(/^#\s+(.+)$/m);
  if (!h1Match) {
    return null;
  }

  const h1Title = h1Match[1].trim();

  // If H1 looks like a prompt, try to find a better title
  if (looksLikePrompt(h1Title)) {
    return extractBetterTitle(content, h1Title);
  }

  return h1Title;
}

/**
 * Extract title from YAML frontmatter (for Knowledge Base - clean Professor Gemini output)
 */
function extractTitleFromFrontmatter(content) {
  // Check for YAML frontmatter
  const frontmatterMatch = content.match(/^---\s*\n([\s\S]*?)\n---/);
  if (frontmatterMatch) {
    const frontmatter = frontmatterMatch[1];
    const titleMatch = frontmatter.match(/^title:\s*["']?(.+?)["']?\s*$/m);
    if (titleMatch) {
      return titleMatch[1].trim();
    }
  }

  // Fallback: first H1
  const h1Match = content.match(/^#\s+(.+)$/m);
  return h1Match ? h1Match[1].trim() : null;
}

/**
 * Extract date from frontmatter or file stats
 */
function extractDateFromFrontmatter(content, stats) {
  const frontmatterMatch = content.match(/^---\s*\n([\s\S]*?)\n---/);
  if (frontmatterMatch) {
    const frontmatter = frontmatterMatch[1];
    const dateMatch = frontmatter.match(/^generated_at:\s*["']?(.+?)["']?\s*$/m);
    if (dateMatch) {
      // Parse "2026-01-15 12:17:37" format
      const dateStr = dateMatch[1].split(' ')[0];
      return dateStr;
    }
  }
  return stats.mtime.toISOString().split('T')[0];
}

/**
 * Clean markdown content
 */
function cleanContent(content) {
  let cleaned = content;

  // Remove Perplexity logo image tags (HTML img tags)
  cleaned = cleaned.replace(/<img[^>]*perplexity[^>]*>/gi, '');

  // Remove any leading empty lines after cleanup
  cleaned = cleaned.replace(/^\s*\n+/, '');

  return cleaned;
}

/**
 * Remove YAML frontmatter from content for display
 */
function removeYamlFrontmatter(content) {
  return content.replace(/^---\s*\n[\s\S]*?\n---\s*\n/, '');
}

/**
 * Process a single markdown file
 */
function processFile(filePath, config) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const stats = fs.statSync(filePath);

  const cleanedContent = cleanContent(content);
  const title = config.titleExtractor(cleanedContent);

  if (!title) {
    console.warn(`  Warning: No title found in ${path.basename(filePath)}, skipping...`);
    return null;
  }

  const slug = generateSlug(title);

  // Use frontmatter date for Knowledge Base, file date for Scratch Pad
  const date = config.name === 'Knowledge Base'
    ? extractDateFromFrontmatter(cleanedContent, stats)
    : stats.mtime.toISOString().split('T')[0];

  // For Knowledge Base, remove frontmatter from displayed content
  const displayContent = config.name === 'Knowledge Base'
    ? removeYamlFrontmatter(cleanedContent)
    : cleanedContent;

  return {
    slug,
    title,
    date,
    content: displayContent,
    sourceFile: path.basename(filePath),
  };
}

/**
 * Generate TypeScript file for a source
 */
function generateTypeScriptFile(documents, config) {
  return `/**
 * ${config.name} Documents
 *
 * Auto-generated by scripts/sync-nebula-docs.js
 * Source: ${config.sourceDir}
 * Generated: ${new Date().toISOString()}
 *
 * DO NOT EDIT MANUALLY - Run "npm run sync:nebula" to regenerate
 */

export interface ${config.interfaceName} {
  slug: string;
  title: string;
  date: string;
  content: string;
  sourceFile: string;
}

export const ${config.exportName}: ${config.interfaceName}[] = ${JSON.stringify(documents, null, 2)};

export function ${config.getterName}(slug: string): ${config.interfaceName} | undefined {
  return ${config.exportName}.find(doc => doc.slug === slug);
}

export function ${config.slugsGetterName}(): string[] {
  return ${config.exportName}.map(doc => doc.slug);
}
`;
}

/**
 * Sync a single source
 * Returns { count: number, files: string[] } for tracking synced files
 */
function syncSource(config) {
  console.log(`\nSyncing ${config.name}...`);
  console.log(`Source: ${config.sourceDir}`);
  console.log(`Output: ${config.outputFile}\n`);

  // Check if source directory exists
  if (!fs.existsSync(config.sourceDir)) {
    console.log(`Source directory not found: ${config.sourceDir}`);
    console.log('Skipping sync - using existing data file (CI/CD environment)');
    return { count: 0, files: [] };
  }

  let files;
  try {
    files = fs.readdirSync(config.sourceDir);
  } catch (error) {
    if (error.code === 'EACCES' || error.code === 'EPERM') {
      console.log(`Access denied for source directory: ${config.sourceDir}`);
      console.log('Skipping sync - using existing data file');
      return { count: 0, files: [] };
    }
    throw error;
  }

  // Get all markdown files
  files = files
    .filter(f => f.endsWith('.md'))
    .map(f => path.join(config.sourceDir, f));

  console.log(`Found ${files.length} markdown files\n`);

  // Process each file
  const documents = [];
  const syncedFiles = [];
  for (const file of files) {
    console.log(`Processing: ${path.basename(file)}`);
    const doc = processFile(file, config);
    if (doc) {
      documents.push(doc);
      syncedFiles.push(file);
      console.log(`  -> "${doc.title}" (${doc.date})`);
    }
  }

  // Sort by date (newest first)
  documents.sort((a, b) => new Date(b.date) - new Date(a.date));

  // Ensure data directory exists
  const dataDir = path.dirname(config.outputFile);
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }

  // Write output file
  const output = generateTypeScriptFile(documents, config);
  fs.writeFileSync(config.outputFile, output, 'utf-8');

  console.log(`\nGenerated ${config.outputFile}`);
  console.log(`Total documents: ${documents.length}`);

  return { count: documents.length, files: syncedFiles };
}

/**
 * Main sync function
 */
function sync() {
  console.log('='.repeat(60));
  console.log('Syncing Nebula Documents');
  console.log('='.repeat(60));

  let totalDocs = 0;
  let knowledgeBaseFiles = [];

  // Sync both sources
  for (const [key, config] of Object.entries(SOURCES)) {
    const result = syncSource(config);
    totalDocs += result.count;

    // Track Knowledge Base files for diagram enhancement
    if (key === 'knowledgeBase') {
      knowledgeBaseFiles = result.files;
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log(`Total documents synced: ${totalDocs}`);
  console.log('='.repeat(60));

  // Signal diagram enhancement needed for Knowledge Base files
  if (knowledgeBaseFiles.length > 0) {
    console.log('\n' + '─'.repeat(60));
    console.log('📊 DIAGRAM ENHANCEMENT PENDING');
    console.log('─'.repeat(60));
    console.log(`${knowledgeBaseFiles.length} Knowledge Base files need diagram review:`);
    knowledgeBaseFiles.forEach(f => console.log(`  • ${path.basename(f)}`));
    console.log('\nClaude Code: Analyze each file and add Mermaid diagrams inline');
    console.log('where they enhance understanding at Principal TPM level.');
    console.log('After adding diagrams, re-run sync to update TypeScript files.');
    console.log('─'.repeat(60));
  }
}

// Run sync
sync();
