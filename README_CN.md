# UQPyL: 参数不确定性分析及优化工具包

<p align="center"><img src="./docs/UQ.svg" width="400"/></p>

[![PyPI version](https://badge.fury.io/py/UQPyL.svg?icon=si%3Apython&icon_color=%2331aadd)](https://badge.fury.io/py/UQPyL) ![PyPI - Downloads](https://img.shields.io/pypi/dm/UQPyL) ![PyPI - License](https://img.shields.io/pypi/l/UQPyL) ![GitHub last commit](https://img.shields.io/github/last-commit/smasky/UQPyL) ![Static Badge](https://img.shields.io/badge/Author-wmtSky-orange) ![Static Badge](https://img.shields.io/badge/Contact-wmtsmasky%40gmail.com-blue)

UQPyL 是一个功能全面的 Python 工具包，专注于参数不确定性分析与优化，适用于数值模型校准、资源调度、产品设计等工程优化问题。该工具包同时集成了多种常用方法，包括实验设计 (Design of Experiments)、敏感性分析 (Sensitivity Analysis) 以及参数优化 (支持单目标与多目标) 。此外，内置的替代模型 (Surrogate Models) 模块可用于计算代价昂贵问题 (Computational Expensive Problem) 的求解。

👉[English Doc](./README.md)

## Contents

- [功能特点](#功能特点)
- [安装指南](#️-installation)
- [链接](#-useful-links)
- [Overview of Methods, Algorithms and Problem](#-overview-of-methods-algorithms-and-problems)
   - [Sensitivity Analysis](#sensitivity-analysis)
   - [Optimization Algorithms](#optimization-algorithms)
   - [Surrogate Models](#surrogate-models)
   - [Single-objective Problems](#single-objective-problems)
   - [Multi-objective Problems](#multi-objective-problems)
- [Quick Start](#-quick-start)
- [Call for Contributions](#-call-for-contributions)
- [Contact](#-contact)

## 功能特点
1. **全面支持敏感性分析与优化**: 实现了目前广泛使用的敏感性分析方法和优化算法。
2. **运行显示与结果保存**: 允许用户跟踪并保存运行历史和结果。
3. **先进的替代模型**: 集成了多种替代模型及自动调优工具，以提升模型性能。
4. **丰富的应用资源**: 提供了全面的基准问题和实际案例，帮助用户快速上手。(👉 近期规划： 针对水科学研究，我们计划定制特定模型专用的程序接口，将水利相关模型与 UQPyL 集成，提升可用性和功能性，类似于我们已开发的[SWAT-UQ](https://github.com/smasky/SWAT-UQ)。如果您感兴趣，欢迎联系我们进行合作。)
5. **模块化与可扩展的架构**: 设计了统一的敏感性分析与优化架构，支持用户快速开发新方法或算法(我们非常欢迎并感谢您对UQPyL的贡献)。
## 安装指南





