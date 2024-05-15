---
layout: post
title: "Hyperjudge — Source-checking tool for doc validity"
description: A web app that checks the validity of linked sources in documents and webpages.
tags: project seo writing tools
image: /img/seo/hyperjudge-cover.png
thumb: /img/thumb/hyperjudge-cover.webp
---

![Hyperjudge Screenshot](/murtohilali.github.io/img/articles/hyperjudge-screenshot.png)

## Table of Contents
- [Overview](#overview)
- [The Plan, Problem, & Solution](#the-plan-problem--solution)
- [Lessons & Takeaways](#lessons--takeaways)
- [Roadmap](#roadmap)


## Overview
Earlier this year, I did editing work for my friend [Tameem](https://twitter.com/TheSaastronaut)'s marketing agency, [TalktheTalk Creative](https://www.wetalkthetalk.co/). I had a few editing tasks associated with this:

- Checking spelling and grammar
- Splitting up extensive paragraphs
- Ensuring factual correctness

Part of that last bullet inlcudes checking the sources an articles links to and ensuring they're:

1. From reputable sources.
2. The fact/stat/figure linked was congruent with the source.

Bullet #2 is a bit trickier to automate (I'll probably need to use an LLM) but #1 is more straightforward.

## The Plan, Problem, & Solution
In essence, all I needed to do was:

- Extract all the hyperlinks and associated text from the body of the page.
- Get the authority score from the domain via Moz API.
- Return the DA scores and average graphically.

However, there was one problem — my ultimate goal is to ship this as a Chrome extension. That means it needs to be written in JavaScript. One problem: the only JavaScript I know is `console.log('Hello, World!)`.

Luckily, I have access to GPT-4o. Combining a high-fidelity mockup:

![](/murtohilali.github.io/img/articles/hyperjudge-hi-fi-mockup.png)

Along with with lots of debugging and iced coffee led to the birth of [Hyperjudge](https://hyperjudge.com):

![Hyperjudge Screenshot](/murtohilali.github.io/img/articles/hyperjudge-screenshot.png)

I was able to build this under a day — I'm sure there are some cracked engineers who could do it an hour or two, but as someone with close to 0 experience with anything other than Python, I'm pretty proud of myself!

## Lessons & Takeaways
- **LLMs are superpowers — but only for those who know how to use them.** Trying to build this project without GPT-4 would probably have taken me days, even weeks. Because of OpenAI (thank you Sam) I could shrink that down to < 24 hours. However, debugging would have been a nightmare if I had absolutely 0 familiarity with programming. LLMs are never going to give you perfect, ship-ready code out of the box, so knowing fundamentals is crucial for building things that work.
- **It's much easier to build side projects as gifts.** One of my goals with Hyperjudge was to help automate one of the tasks I used to execute for Tameem — since I had a user in mind, it was a lot easier to stay motivated and on track. 

## Roadmap

1. A Chrome extension that automatically checks the source validity of a page or Google Doc.
2. Add a feature that checks to ensure the hyperlinked text aligns with the source. (For instance, the source says, '2 out of 3 people love Murto's blog posts on Medium' and the hyperlink says '67% of readers enjoy Murto's blog posts').