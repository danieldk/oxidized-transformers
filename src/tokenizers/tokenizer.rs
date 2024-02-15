use crate::error::BoxedError;

use snafu::Snafu;

use super::pieces::PiecesWithIds;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(super)))]
pub enum TokenizerError {
    #[snafu(display("Couldn't encode tokenizer inputs into pieces and ids"))]
    Encode { source: tokenizers::Error },

    #[snafu(display("Couldn't decode piece identifiers into strings"))]
    Decode { source: tokenizers::Error },

    #[snafu(display("HuggingFace tokenizer error"))]
    HFTokenizer { source: BoxedError },
}

pub enum TokenizerEncodeInput<I>
where
    I: AsRef<str>,
{
    RawString(I),
}

impl From<String> for TokenizerEncodeInput<String> {
    fn from(s: String) -> Self {
        TokenizerEncodeInput::RawString(s)
    }
}

/// Trait implemented by all tokenizers.
pub trait Tokenizer {
    /// Type of the piece identifiers.
    type PieceId: Copy + Eq + Ord;

    /// Split one or more texts into pieces.
    ///
    /// * input - Sequences to tokenize. If the sequences are
    ///   strings, they are automatically converted to chunks.
    ///
    /// Returns: Pieces in each sequence.
    fn encode<V, I>(&self, input: V) -> Result<PiecesWithIds<Self::PieceId>, TokenizerError>
    where
        V: AsRef<[TokenizerEncodeInput<I>]>,
        I: AsRef<str>;

    /// Reconstruct string sequences from piece identifiers.
    ///
    /// * input - The piece identifiers to reconstruct the strings from.
    /// * skip_special_pieces - Skip special pieces during decoding.
    ///
    /// Returns: The decoded strings.
    fn decode<V, I>(
        &self,
        input: V,
        skip_special_pieces: bool,
    ) -> Result<Vec<String>, TokenizerError>
    where
        V: AsRef<[I]>,
        I: AsRef<[Self::PieceId]>;

    /// Get the ID for a single piece.
    ///
    /// * piece - The piece to look up the identifier for.
    ///
    /// Returns: The piece identifier, `None` when the piece
    /// is unknown.
    fn piece_to_id(&self, piece: impl AsRef<str>) -> Option<Self::PieceId>;

    /// Get the end-of-sequence piece.
    ///
    /// Returns: The end-of-sequence piece or
    /// `None` when this piece is not defined.
    fn eos_piece(&self) -> Option<&str>;
}
