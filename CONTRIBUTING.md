# Contributing to libpsam

Thank you for your interest in contributing to libpsam! This document provides guidelines and information for contributors.

## Ways to Contribute

- 🐛 **Bug Reports** - Report issues you encounter
- 💡 **Feature Requests** - Suggest new features or improvements
- 📝 **Documentation** - Improve docs, examples, or guides
- 🔧 **Code** - Submit bug fixes or new features
- 🧪 **Testing** - Test on different platforms or use cases
- 💬 **Discussion** - Share ideas and experiences

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/Foundation42/libpsam.git
   cd libpsam
   ```

2. **Build the project**
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_EXAMPLES=ON ..
   cmake --build .
   ```

3. **Run examples to verify**
   ```bash
   ./examples/c/basic_usage
   ```

## Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Follow existing code style
   - Add tests if applicable
   - Update documentation

3. **Test your changes**
   ```bash
   # Build and test
   cmake --build .
   ./examples/c/basic_usage
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### C Code

- **Standard**: C11
- **Indentation**: 4 spaces (no tabs)
- **Naming**:
  - Functions: `psam_function_name`
  - Types: `psam_type_name_t`
  - Constants: `PSAM_CONSTANT_NAME`
- **Comments**: Use `/* */` for multi-line, `//` for single-line
- **Headers**: Include guards for all headers

Example:
```c
/**
 * Brief description of function
 *
 * @param model Model handle
 * @param token Token ID
 * @return Error code
 */
psam_error_t psam_train_token(psam_model_t* model, uint32_t token) {
    if (!model) {
        return PSAM_NULL_PARAM;
    }

    // Implementation
    return PSAM_OK;
}
```

### JavaScript/TypeScript

- **Style**: Follow existing patterns
- **Formatting**: 2-space indentation
- **Types**: Use TypeScript types where possible
- **Naming**: camelCase for functions, PascalCase for classes

### Python

- **Style**: PEP 8
- **Type hints**: Use where helpful
- **Docstrings**: Google style
- **Formatting**: Black (if available)

Example:
```python
def train_batch(self, tokens: List[int]) -> None:
    """
    Process a batch of tokens during training.

    Args:
        tokens: List of token IDs to process

    Raises:
        PSAMError: If training fails
    """
    # Implementation
```

## Testing

### C Library

Add tests in `core/src/` (we'll expand test infrastructure):

```c
void test_basic_training() {
    psam_model_t* model = psam_create(100, 8, 10);
    assert(model != NULL);

    uint32_t tokens[] = {1, 2, 3, 4, 5};
    psam_error_t err = psam_train_batch(model, tokens, 5);
    assert(err == PSAM_OK);

    psam_destroy(model);
}
```

### Language Bindings

- **JavaScript**: Add tests using Vitest
- **Python**: Add tests using pytest

## Documentation

- Update README files when adding features
- Add examples for new functionality
- Update API documentation in `docs/API.md`
- Include code comments for complex logic

## Commit Messages

Format:
```
type(scope): brief description

Longer description if needed.

- Bullet points for details
- Reference issues: Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Build, CI, etc.

Examples:
```
feat(core): add support for custom distance decay functions

fix(python): handle null predictions correctly

docs(readme): add WASM build instructions
```

## Pull Request Process

1. **Ensure builds pass** on your platform
2. **Update documentation** for user-facing changes
3. **Add examples** for new features
4. **Describe changes** clearly in PR description
5. **Reference issues** if applicable (Fixes #123)
6. **Be responsive** to review feedback

## Areas We'd Love Help With

### Core Library
- ✅ Performance optimizations (SIMD, parallelization)
- ✅ Additional weighting schemes beyond PPMI/IDF
- ✅ Memory pool allocators
- ✅ Comprehensive test suite

### Language Bindings
- ✅ Additional languages (Rust, Go, Ruby, etc.)
- ✅ N-API bindings for Node.js (currently FFI-only)
- ✅ Improved WASM interface
- ✅ Better error messages

### Documentation
- ✅ Tutorials and guides
- ✅ More examples
- ✅ Performance benchmarks
- ✅ Video tutorials

### Applications
- ✅ Real-world use cases
- ✅ Integration examples (web frameworks, etc.)
- ✅ Pre-trained models for common tasks
- ✅ Comparison with other approaches

## Questions?

- **GitHub Issues**: https://github.com/Foundation42/libpsam/issues
- **Discussions**: https://github.com/Foundation42/libpsam/discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to libpsam! 🎉
